import argparse
import json
import os
import re
import sys
from getpass import getpass
from pathlib import Path

import psycopg
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from urllib.error import URLError

from .db import Store, passport_key
from .graph import ISO_DATE, LABELLED_PASSPORT, PASSPORT, Agent, new_thread
from .model import Ollama

HELP = """Вход и регистрация
  /register ЛОГИН  · создать аккаунт
  /login ЛОГИН     · войти
  /logout          · выйти из аккаунта

Диалог
  /passport        · подтвердить бронь паролем, затем ввести паспорт
  /bookings        · показать свои демоброни
  /cancel          · подтвердить отмену; /cancel ID выбирает бронь
  /reset           · очистить память диалога
  /trace           · показать или скрыть вызовы инструментов
  /graph           · вывести граф в формате Mermaid
  /help            · показать команды
  /exit            · закрыть консоль"""


class ConsoleHistory(InMemoryHistory):
    def append_string(self, string):
        line = string.strip()
        if not line or PASSPORT.search(line) or LABELLED_PASSPORT.search(line):
            return
        if re.fullmatch(r"[0-9 -]+", line) and not ISO_DATE.fullmatch(line):
            return
        command, *args = line.split()
        if (command in {"/login", "/register"} and len(args) > 1) or (command == "/passport" and args):
            return
        super().append_string(string)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", action="store_true", help="Вывести граф Mermaid и завершить работу")
    options = parser.parse_args()
    if options.graph:
        print(Agent(None, None, b"").graph.get_graph().draw_mermaid())
        return
    console = Console()
    if not sys.stdin.isatty():
        raise SystemExit("Для ввода скрытых данных запустите агент в интерактивном терминале.")
    store = Store()
    try:
        store.initialize()
        key = passport_key(Path(os.getenv("PASSPORT_KEY_FILE", "passport.key")))
    except (psycopg.Error, OSError, ValueError):
        raise SystemExit("Не удалось подключить PostgreSQL или открыть ключ паспортов. Проверьте настройки Compose.")
    agent = Agent(store, Ollama(), key)
    account, thread, trace_enabled = None, new_thread(), False
    session = PromptSession(history=ConsoleHistory())
    console.print(Panel("Учебное расписание · демобронирование\nНачните с /register или /login. Список команд: /help.",
                        title="✈  Авиаагент", border_style="cyan"))
    while True:
        try:
            prompt = account.username if account else "Гость"
            line = session.prompt([("bold ansicyan", f"{prompt} › ")]).strip()
            if not line:
                continue
            command, _, value = line.partition(" ")
            value = value.strip()
            if command == "/exit":
                break
            if command == "/help":
                console.print(HELP)
            elif command == "/graph":
                console.print(Text(agent.graph.get_graph().draw_mermaid()))
            elif command in {"/register", "/login"}:
                if account:
                    console.print("Сначала завершите текущий сеанс через /logout.")
                    continue
                if not value or any(c.isspace() for c in value):
                    console.print(f"Формат команды: {command} ЛОГИН. Пароль вводится в отдельном скрытом запросе.")
                    continue
                password = getpass("Пароль: ")
                try:
                    if command == "/register":
                        username = " ".join(console.input("Фамилия Имя: ").split())
                        registered = store.register(value.lower(), password, username)
                        console.print("Аккаунт создан. Войдите через /login." if registered else "Логин уже занят.")
                    else:
                        account = store.authenticate(value.lower(), password)
                        console.print("Вы вошли." if account else "Неверный логин или пароль.")
                finally:
                    del password
            elif command == "/logout":
                agent.memory.delete_thread(thread)
                account, thread, trace_enabled = None, new_thread(), False
                session = PromptSession(history=ConsoleHistory())
                console.print("Сеанс завершён.")
            elif not account:
                console.print("Войдите через /login или создайте аккаунт через /register.")
            elif command == "/reset":
                agent.memory.delete_thread(thread)
                thread = new_thread()
                console.print("Диалог очищен.")
            elif command == "/trace":
                trace_enabled = not trace_enabled
                console.print(f"Трассировка {'включена' if trace_enabled else 'выключена'}.")
            elif command in {"/passport", "/bookings", "/cancel"} or not command.startswith("/"):
                if command == "/cancel" and not value:
                    answer, calls = agent.confirm_cancellation(thread, account, getpass)
                elif command == "/passport":
                    if value:
                        console.print("Введите /passport без аргументов, затем укажите номер в скрытом запросе.")
                        continue
                    answer, calls = agent.confirm_booking(thread, account, getpass)
                elif command.startswith("/"):
                    answer, calls = agent.ask(thread, account, line)
                else:
                    with console.status("Qwen анализирует сообщение…", spinner="dots"):
                        answer, calls = agent.ask(thread, account, line)
                if trace_enabled:
                    trace = json.dumps(calls, ensure_ascii=False) if calls else "инструмент не вызван"
                    console.print(Text(trace, style="dim"))
                response = "\n".join([answer["message"], *(f"• {item}" for item in answer["alternatives"])])
                console.print(Panel(Text(response), title="Авиаагент", border_style="blue"))
            else:
                console.print("Неизвестная команда. Список команд: /help.")
        except (EOFError, KeyboardInterrupt):
            break
        except (URLError, TimeoutError, KeyError, ValueError) as exc:
            message = str(exc) if type(exc) is ValueError and command == "/register" else (
                "Не удалось получить корректное решение от Ollama. Проверьте доступность модели и повторите запрос.")
            console.print(Text(message, style="yellow"))
        except psycopg.Error:
            console.print("[yellow]Ошибка PostgreSQL. Повторите запрос после восстановления базы.[/]")


if __name__ == "__main__":
    main()
