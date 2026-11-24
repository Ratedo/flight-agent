import hashlib
import hmac
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from .data import FLIGHTS

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    login text UNIQUE NOT NULL,
    username text NOT NULL,
    salt bytea NOT NULL,
    password_hash bytea NOT NULL
);
CREATE TABLE IF NOT EXISTS flights (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    number text NOT NULL,
    airline text NOT NULL,
    origin text NOT NULL,
    destination text NOT NULL,
    flight_date date NOT NULL,
    departure time NOT NULL,
    status text NOT NULL,
    seats integer NOT NULL CHECK (seats >= 0),
    UNIQUE (number, flight_date)
);
CREATE TABLE IF NOT EXISTS bookings (
    id uuid PRIMARY KEY,
    account_id bigint NOT NULL REFERENCES accounts(id),
    flight_id bigint NOT NULL REFERENCES flights(id),
    passenger_name text NOT NULL,
    passport_digest bytea NOT NULL
);
ALTER TABLE flights ADD COLUMN IF NOT EXISTS price_light integer CHECK (price_light > 0);
ALTER TABLE flights ADD COLUMN IF NOT EXISTS price_baggage integer CHECK (price_baggage >= price_light);
ALTER TABLE bookings ADD COLUMN IF NOT EXISTS fare_price integer CHECK (fare_price > 0);
ALTER TABLE bookings ADD COLUMN IF NOT EXISTS fare_baggage boolean;
ALTER TABLE bookings ADD COLUMN IF NOT EXISTS cancelled_at timestamptz;
CREATE UNIQUE INDEX IF NOT EXISTS active_booking_per_passport
    ON bookings (flight_id, passport_digest) WHERE cancelled_at IS NULL;
ALTER TABLE bookings DROP CONSTRAINT IF EXISTS bookings_flight_id_passport_digest_key;
"""


@dataclass(frozen=True)
class Account:
    id: int
    username: str


def valid_name(name: str) -> bool:
    return len(name) <= 100 and bool(re.fullmatch(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*(?: [^\W\d_]+(?:[-’'][^\W\d_]+)*){1,3}", name))


def password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**15, r=8, p=1, maxmem=64 * 1024**2)


def passport_key(path: Path) -> bytes:
    if not path.exists():
        # Link a complete key atomically, including when two consoles start together.
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}")
        try:
            with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as handle:
                handle.write(secrets.token_bytes(32))
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
    key = path.read_bytes()
    if len(key) != 32:
        raise ValueError("Неверный размер ключа паспортов")
    return key


class Store:
    def __init__(self, dsn: str = ""):
        self.dsn = dsn

    def connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=5)

    def initialize(self):
        with self.connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(724015)")
            cursor.execute(SCHEMA)
            cursor.executemany(
                """INSERT INTO flights (number, airline, origin, destination, flight_date, departure, status,
                                        seats, price_light, price_baggage)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (number, flight_date) DO UPDATE SET
                       price_light = COALESCE(flights.price_light, EXCLUDED.price_light),
                       price_baggage = COALESCE(flights.price_baggage, EXCLUDED.price_baggage)
                   WHERE flights.price_light IS NULL OR flights.price_baggage IS NULL""",
                FLIGHTS.itertuples(index=False, name=None),
            )
            cursor.execute("ALTER TABLE flights ALTER COLUMN price_light SET NOT NULL, "
                           "ALTER COLUMN price_baggage SET NOT NULL")

    def register(self, login: str, password: str, username: str) -> bool:
        if not re.fullmatch(r"[a-z0-9_.-]{3,32}", login):
            raise ValueError("Логин: от 3 до 32 символов a–z, 0–9, точка, дефис или подчёркивание.")
        if not 8 <= len(password) <= 128:
            raise ValueError("Пароль должен содержать от 8 до 128 символов.")
        if not valid_name(username):
            raise ValueError("Укажите фамилию и имя через пробел.")
        salt = secrets.token_bytes(16)
        with self.connect() as connection:
            row = connection.execute(
                """INSERT INTO accounts (login, username, salt, password_hash) VALUES (%s, %s, %s, %s)
                   ON CONFLICT (login) DO NOTHING RETURNING id""",
                (login, username, salt, password_hash(password, salt)),
            ).fetchone()
        return row is not None

    def authenticate(self, login: str, password: str) -> Account | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT id, username, salt, password_hash FROM accounts WHERE login = %s", (login,),
            ).fetchone()
        if self._matches_password(row, password):
            return Account(row["id"], row["username"])
        return None

    def verify_password(self, account_id: int, password: str) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT salt, password_hash FROM accounts WHERE id = %s", (account_id,),
            ).fetchone()
        return self._matches_password(row, password)

    @staticmethod
    def _matches_password(row, password):
        if len(password) > 128:
            return False
        supplied = password_hash(password, row["salt"] if row else b"\0" * 16)
        return row is not None and hmac.compare_digest(supplied, row["password_hash"])
