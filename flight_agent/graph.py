import hmac
import re
from datetime import date
from decimal import Decimal
from difflib import SequenceMatcher
from typing import TypedDict
from urllib.error import URLError
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .data import AIRLINES, CITIES
from .db import valid_name
from .tools import (book_passengers, cancel_booking, check_booking, describe, describe_fare, find_cancellable,
                    flight_status, list_flights, my_bookings, result)

PASSPORT = re.compile(r"(?<!\d)(?:[0-9]{10}|[0-9]{2}[ -]?[0-9]{2}[ -][0-9]{6})(?!\d)")
LABELLED_PASSPORT = re.compile(r"(?i)\b(?:номер\s+паспорта|паспорт(?:\s+РФ)?)\s*(?:[:=№]\s*|\s+(?=\d))([^;,\n]+)")
ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
FLIGHT = re.compile(r"[A-Z0-9]{2}[0-9]{1,4}")
WITHOUT_PASSPORT = re.compile(r"(?i)\bбез\s+паспорта\b")
BUY_VERB = r"купи(?:ть|те|м)?|покупа(?:й(?:те)?|ть|ем)"
BOOKING_WORD = re.compile(rf"(?i)\b(?:заброни\w*|брон[ьи]\w*|билет\w*|{BUY_VERB})\b")
MY_BOOKINGS = re.compile(r"(?i)\b(?:мои|свои)\s+(?:брони|бронирования|билеты)\b")
CANCEL = re.compile(r"(?i)\b(?:удал(?:и\w*|ен\w*)|отмен(?:и\w*|яй\w*|а|у)|аннулир\w*)\b")
BOOKING_ID = re.compile(r"(?i)\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b")
PASSPORT_OR_BOOKING_ID = re.compile(f"{BOOKING_ID.pattern.removeprefix('(?i)')}|{PASSPORT.pattern}", re.I)
BOOK_SELECTED = re.compile(r"(?i)(?:давай\s+)?заброниру(?:й|ем)\s+(?:этот рейс|его)[.!]?")
FIELDS = ("number", "flight_date", "airline", "origin", "destination")
FARE_FIELDS = ("fare_price", "fare_baggage")
PRICE = re.compile(
    r"(?i)(?:\bза\s+|(?P<upto>\bдо\s+)|^)(?:(?P<k>[0-9]{1,4}(?:[,.][0-9]{1,3})?)\s*(?:[кk]|тысяч(?:у|и)?|тыс\.?)"
    r"|(?P<rub>[0-9]{1,4}(?:[ \u00a0\u202f][0-9]{3})+|[0-9]{4,7}))(?![\w.,-])")
TAKE = re.compile(rf"(?i)\b(?:давай\w*|беру|бери\w*|возьм\w*|оформ\w*|заброни\w*|{BUY_VERB})\b")
BUY_POSITION = re.compile(
    rf"(?i)\b(?:{BUY_VERB}|возьми(?:те)?|оформи(?:те)?|забронируй(?:те)?)"
    r"\s+(?:рейс\s+)?№?\s*([1-9][0-9]?)(?!\w|[.,:-]\d|\s+(?:билет|мест)\w*)")
BARE_PURCHASE = re.compile(
    rf"(?i)(?:(?:давай\s+)?(?:{BUY_VERB}|бери|беру|возьми|оформи|оформляй|забронируй)"
    r"(?:\s+(?:его|этот рейс|билет))?|давай\s+(?:его|этот рейс))[.!]?")
BARE_FARE = re.compile(r"(?i)(?:давай\s+)?(?:без багажа|с багажом)[.!]?")
SMALL_TALK = re.compile(r"(?i)(?:хай|привет(?:ик)?|здравствуй(?:те)?|добрый (?:день|вечер)|как дела)[.!?,\s]*")
CHOICE = re.compile(r"(?i)(?:давай\s+)?вариант\s*№?\s*([0-9]+)[.!]?")
ORDINALS = ("перв", "втор", "трет", "четверт", "пят", "шест", "седьм", "восьм")
ORDINAL = re.compile(r"\b(" + "|".join(ORDINALS) + r")(?:ый|ой|ий|ю|ую|ая|ья|ого|его|ьего|ьим)\b")
DEPARTURE_TIME = re.compile(r"\b([01]?[0-9]|2[0-3]):([0-5][0-9])\b")
MENTIONED_FLIGHT = re.compile(r"\b(?=[A-Z0-9]*[A-Z])[A-Z0-9]{2}[0-9]{1,4}\b")
PETERSBURG_ONLY = re.compile(r"(?i)(?:мне\s+нужно|хочу|лечу|го)\s+(?:лететь\s+)?в\s+(?:питер|петербург|спб)[.!?]?")
CHEAPEST = re.compile(r"\b(?:дешевле|дешев(?:ый|ая|ое|ого)|недорогой)\b")
CHEAPEST_WITH_TYPO = re.compile(r"\bсам(?:ый|ая|ое|ого)\s+(деш[а-я]{2,5})\b")
QUANTITIES = {"один": 1, "одно": 1, "два": 2, "две": 2, "три": 3, "четыре": 4,
              "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10}
TICKET_COUNT = re.compile(r"(?i)\b([1-9][0-9]{0,2}|" + "|".join(QUANTITIES) + r")\s+(?:билет\w*|мест\w*)\b")
MONTHS = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
          "сентября", "октября", "ноября", "декабря")
RUSSIAN_DATE = re.compile(r"(?i)\b([0-9]{1,2})\s+(" + "|".join(MONTHS) + r")(?:\s+([0-9]{4}))?\b")
ISO_DATE_IN_TEXT = re.compile(r"\b" + ISO_DATE.pattern + r"\b")


class AgentState(TypedDict, total=False):
    text: str
    account_id: int
    username: str
    passengers: list[tuple] | None
    selected: dict
    pending: str
    draft: dict
    intent: str
    arguments: dict
    node: str
    trace: list
    answer: dict
    model_reply: str | None
    booking_account: int | None
    recent_offers: list[dict]
    offer_choices: list[dict]
    offer_quantity: int
    cancellation: dict | None


def clean_action(action):
    fields = {key: value.strip().upper() for key in FIELDS if (value := getattr(action, key))}
    if not FLIGHT.fullmatch(fields.get("number", "")):
        fields.pop("number", None)
    for key, allowed in (("airline", AIRLINES), ("origin", CITIES), ("destination", CITIES)):
        if fields.get(key) not in allowed:
            fields.pop(key, None)
    if "flight_date" in fields:
        try:
            fields["flight_date"] = date.fromisoformat(fields["flight_date"]).isoformat()
        except ValueError:
            fields.pop("flight_date")
    return fields


def take_offers(answer):
    if "offers" in answer:
        return {"recent_offers": answer.pop("offers"), "offer_choices": []}
    return {}


def match_recent_offers(text, offers):
    normalized = text.lower().replace("ё", "е")
    price = PRICE.search(normalized)
    ordinal = ORDINAL.search(normalized)
    position = BUY_POSITION.search(normalized)
    departure = DEPARTURE_TIME.search(normalized)
    cheapest = bool(CHEAPEST.search(normalized))
    if not cheapest and (approximate := CHEAPEST_WITH_TYPO.search(normalized)):
        cheapest = SequenceMatcher(None, approximate[1], "дешевый").ratio() >= 0.8
    baggage = None
    if re.search(r"\bбез багажа\b", normalized):
        baggage = False
    elif re.search(r"\bс багажом\b", normalized):
        baggage = True
    if not (price or ordinal or position or departure or cheapest or baggage is not None):
        return None
    if baggage is not None:
        offers = [offer for offer in offers if offer["fare_baggage"] == baggage]
    if flight := MENTIONED_FLIGHT.search(text.upper()):
        offers = [offer for offer in offers if offer["number"] == flight[0]]
    if ordinal or position:
        rank = int(position[1]) if position else ORDINALS.index(ordinal[1]) + 1
        offers = [offer for offer in offers if offer["flight_position"] == rank]
    if departure:
        time = f"{int(departure[1]):02}:{departure[2]}"
        offers = [offer for offer in offers if offer["departure"] == time]
    if price:
        amount = (int(Decimal(price["k"].replace(",", ".")) * 1000)
                  if price["k"] else int(re.sub(r"\s", "", price["rub"])))
        offers = [offer for offer in offers
                  if offer["fare_price"] == amount or (price["upto"] and offer["fare_price"] < amount)]
    if cheapest and offers:
        available = [offer for offer in offers if offer["seats"] and offer["status"] != "отменён"]
        if not available:
            return []
        minimum = min(offer["fare_price"] for offer in available)
        offers = [offer for offer in available if offer["fare_price"] == minimum]
    return [offer["offer_id"] for offer in offers]


def offers_on_date(text, offers):
    if iso := ISO_DATE_IN_TEXT.search(text):
        return [offer for offer in offers if offer["flight_date"] == iso[0]]
    if spoken := RUSSIAN_DATE.search(text.lower()):
        suffix = f"-{MONTHS.index(spoken[2]) + 1:02}-{int(spoken[1]):02}"
        return [offer for offer in offers if offer["flight_date"].endswith(suffix)
                and (not spoken[3] or offer["flight_date"].startswith(spoken[3]))]
    return None


class Agent:
    def __init__(self, store, model, key):
        self.store = store
        self.model = model
        self.key = key
        self.memory = InMemorySaver()
        graph = StateGraph(AgentState)
        graph.add_node("route", self.route)
        graph.add_node("gate", self.gate)
        graph.add_node("flight_status", self.status)
        graph.add_node("list_flights", self.listing)
        graph.add_node("my_bookings", self.bookings)
        graph.add_node("find_cancellable", self.find_cancellation)
        graph.add_node("check_booking", self.check)
        graph.add_node("book_ticket", self.book)
        graph.add_node("reply", self.reply)
        graph.add_edge(START, "route")
        graph.add_edge("route", "gate")
        graph.add_conditional_edges("gate", lambda s: s["node"],
                                    ["flight_status", "list_flights", "my_bookings", "find_cancellable", "check_booking",
                                     "book_ticket", "reply"])
        graph.add_edge("flight_status", "reply")
        graph.add_edge("list_flights", "reply")
        graph.add_edge("my_bookings", "reply")
        graph.add_edge("find_cancellable", "reply")
        graph.add_edge("check_booking", "reply")
        graph.add_edge("book_ticket", "reply")
        graph.add_edge("reply", END)
        self.graph = graph.compile(checkpointer=self.memory)

    def ask(self, thread, account, text=""):
        return self._ask(thread, account, text)

    def confirm_booking(self, thread, account, read_secret):
        config = {"configurable": {"thread_id": thread}, "recursion_limit": 8}
        state = self.graph.get_state(config).values
        if state.get("pending") != "book" or state.get("booking_account") != account.id:
            return result("clarify", "Сначала выберите доступный рейс для бронирования."), []
        fields = state["draft"]
        _, problem = check_booking(self.store, fields)
        if problem:
            self.memory.delete_thread(thread)
            take_offers(problem)
            return problem, [{"tool": "check_booking", "args": fields}]
        if not self.store.verify_password(account.id, read_secret("Пароль для подтверждения брони: ")):
            return result("wrong_password", "Неверный пароль. Паспорт не запрошен. Повторите /passport.",
                          missing=["password"]), []
        quantity = fields.get("quantity", 1)
        passengers = []
        for i in range(1, quantity + 1):
            name = account.username if i == 1 else " ".join(read_secret(f"Фамилия и имя пассажира {i}: ").split())
            if i > 1 and not valid_name(name):
                return result("invalid_passenger", f"Укажите фамилию и имя пассажира {i}. Повторите /passport."), []
            prompt = "Демо-паспорт: " if quantity == 1 else f"Демо-паспорт пассажира {i} ({name}): "
            passport = read_secret(prompt).strip()
            digest = (hmac.digest(self.key, re.sub(r"[^0-9]", "", passport).encode(), "sha256")
                      if PASSPORT.fullmatch(passport) else None)
            passengers.append((name, digest))
        return self._ask(thread, account, "", passengers=passengers)

    def confirm_cancellation(self, thread, account, read_secret):
        config = {"configurable": {"thread_id": thread}}
        state = self.graph.get_state(config).values
        selection = state.get("cancellation")
        if state.get("pending") != "cancel" or not selection or selection["account_id"] != account.id:
            return result("clarify", "Сначала укажите бронь для отмены, например «отмени мою бронь S7203»."), []
        if not self.store.verify_password(account.id, read_secret("Пароль для отмены брони: ")):
            return result("wrong_password", "Неверный пароль. Повторите /cancel."), []
        booking_id = selection["booking_id"]
        answer = cancel_booking(self.store, account.id, booking_id)
        self.graph.update_state(config, {"pending": "", "cancellation": None}, as_node="reply")
        return answer, [{"tool": "cancel_booking", "args": {"booking_id": booking_id}}]

    def _ask(self, thread, account, text, *, passengers=None):
        # Public chat never supplies passport data to the booking tool.
        text = text.strip()
        config = {"configurable": {"thread_id": thread}, "recursion_limit": 8}
        text = PASSPORT_OR_BOOKING_ID.sub(
            lambda match: match[0] if BOOKING_ID.fullmatch(match[0]) else "[паспорт скрыт]",
            LABELLED_PASSPORT.sub("[паспорт скрыт]", text),
        )
        if re.fullmatch(r"[0-9 -]+", text) and not ISO_DATE.fullmatch(text) and not PRICE.fullmatch(text):
            previous = self.graph.get_state(config).values
            if previous.get("pending") == "book":
                text = "[паспорт скрыт]"
        state = self.graph.invoke({
            "text": text, "account_id": account.id, "username": account.username, "passengers": passengers,
            "trace": [], "model_reply": None,
        }, config)
        # Keep conversational memory while discarding checkpoint history from this turn.
        remembered = {key: state[key] for key in (
            "selected", "pending", "draft", "booking_account", "recent_offers", "offer_choices", "offer_quantity",
            "cancellation"
        ) if key in state}
        self.memory.delete_thread(thread)
        self.graph.update_state(config, remembered, as_node="reply")
        return state["answer"], state["trace"]

    def route(self, state):
        pending = state.get("pending", "")
        text = state["text"]
        offers = state.get("recent_offers", [])
        choices = state.get("offer_choices", [])
        quantity = {}
        if count := TICKET_COUNT.search(text):
            value = count[1].lower()
            quantity["quantity"] = int(value) if value.isdecimal() else QUANTITIES[value]
        choice_quantity = state.get("offer_quantity", 1)
        carried_quantity = {"quantity": choice_quantity} if choice_quantity > 1 else {}
        converse = getattr(self.model, "converse", None)
        if converse and SMALL_TALK.fullmatch(text):
            return {"intent": "other", "arguments": {}, "model_reply": converse(text)}
        action = error = None
        if text and text != "[паспорт скрыт]" and not text.startswith("/"):
            catalog = [{key: value for key, value in offer.items() if key != "label"} for offer in offers]
            try:
                action = self.model(text, pending, catalog)
            except (URLError, TimeoutError, KeyError, ValueError) as exc:
                error = exc
        booking_id = BOOKING_ID.search(text)
        number = MENTIONED_FLIGHT.search(text.upper())
        if CANCEL.search(text) or text.startswith("/cancel ") or (
            pending == "cancel" and (BOOKING_ID.fullmatch(text) or FLIGHT.fullmatch(text.upper()))
        ):
            fields = {}
            if booking_id:
                fields["booking_id"] = booking_id[0].lower()
            elif number:
                fields["number"] = number[0]
            return {"intent": "cancel", "arguments": fields}
        if state["text"] == "/bookings" or MY_BOOKINGS.search(state["text"]):
            return {"intent": "bookings", "arguments": {}}
        if WITHOUT_PASSPORT.search(state["text"]) and (pending == "book" or BOOKING_WORD.search(state["text"])):
            return {"intent": "passport_required", "arguments": {}}
        if state.get("selected") and BOOK_SELECTED.fullmatch(text):
            return {"intent": "book", "arguments": state["selected"]}
        if not state["text"] or state["text"] == "[паспорт скрыт]":
            return {"intent": pending or "other", "arguments": {}}
        if choices and (choice := CHOICE.fullmatch(state["text"])):
            index = int(choice[1]) - 1
            ids = [choices[index]["offer_id"]] if 0 <= index < len(choices) else []
            return {"intent": "offer_book", "arguments": {"ids": ids, **carried_quantity}}
        if choices and BARE_FARE.fullmatch(text):
            return {"intent": "offer_book", "arguments": {"ids": match_recent_offers(text, choices), **carried_quantity}}
        if BARE_PURCHASE.fullmatch(text):
            if pending == "book" and state.get("booking_account") == state["account_id"]:
                return {"intent": "book", "arguments": {}}
            candidates = choices or offers
            flights = {(offer["number"], offer["flight_date"]) for offer in candidates}
            if len(flights) == 1:
                return {"intent": "offer_book", "arguments": {
                    "ids": [offer["offer_id"] for offer in candidates], **(carried_quantity if choices else {})}}
            if flights:
                return {"intent": "choose_flight", "arguments": {}}
        if offers and not number and TAKE.search(text) and (dated := offers_on_date(text, offers)) is not None:
            ids = match_recent_offers(text, dated)
            return {"intent": "offer_book", "arguments": {
                "ids": ids if ids is not None else [offer["offer_id"] for offer in dated], **quantity}}
        if choices and carried_quantity and not quantity and not number and PRICE.search(text):
            return {"intent": "offer_book", "arguments": {
                "ids": match_recent_offers(text, choices), **carried_quantity}}
        if offers and (ids := match_recent_offers(state["text"], offers)) is not None:
            intent = "offer_book" if TAKE.search(state["text"]) else "offer_info"
            return {"intent": intent, "arguments": {"ids": ids, **quantity}}
        if pending and FLIGHT.fullmatch(state["text"].upper()):
            return {"intent": pending, "arguments": {"number": state["text"].upper()}}
        if pending and ISO_DATE.fullmatch(state["text"]):
            try:
                return {"intent": pending, "arguments": {"flight_date": date.fromisoformat(state["text"]).isoformat()}}
            except ValueError:
                return {"intent": pending, "arguments": {}}
        if PETERSBURG_ONLY.fullmatch(text):
            return {"intent": "list", "arguments": {"destination": "LED"}}
        if error:
            raise error
        if action is None:
            return {"intent": "other", "arguments": {}}
        intent = action.intent
        if intent == "continue":
            intent = pending or "other"
        arguments = {"ids": action.offer_ids} if intent in {"offer_info", "offer_book"} else clean_action(action)
        if intent in {"book", "offer_book"}:
            arguments.update(quantity)
        reply = (action.reply or "").strip()
        if intent == "other" and converse and (not reply or reply == "Чем помочь с рейсами?"):
            reply = converse(text)
        return {"intent": intent, "arguments": arguments, "model_reply": reply}

    def gate(self, state):
        intent = state["intent"]
        if intent == "passport_required":
            return {"node": "reply", "answer": result(
                "clarify", "Для демоброни нужен паспорт. Запросите бронь рейса, затем вызовите /passport.",
                missing=["passport"])}
        if intent == "other":
            message = (state["model_reply"] or "").strip()
            if message:
                return {"node": "reply", "answer": result("chat", message)}
            return {"node": "reply", "answer": result("empty_reply", "Qwen не вернула ответ. Повторите сообщение.")}
        if intent == "list":
            return {"node": "list_flights"}
        if intent == "bookings":
            return {"node": "my_bookings"}
        if intent == "choose_flight":
            return {"node": "reply", "answer": result(
                "clarify", "Какой рейс из последнего списка выбрать? Например, «купи 2».", missing=["number"])}
        if intent == "cancel":
            return {"node": "find_cancellable", "pending": "cancel", "draft": {}, "booking_account": None,
                    "cancellation": None}
        if intent in {"offer_info", "offer_book"}:
            return self.offer_selection(state, book=intent == "offer_book")
        fields = dict(state.get("draft", {})) if state.get("pending") == intent else {}
        new = state["arguments"]
        if any(key in fields and fields[key] != new[key] for key in ("number", "airline", "origin", "destination")
               if key in new):
            fields = {}
        if new and any(fields.get(key) != value for key, value in new.items()):
            for key in FARE_FIELDS:
                fields.pop(key, None)
        if intent == "book" and not fields and not any(key in new for key in ("number", "origin", "destination", "airline")):
            fields.update(state.get("selected") or {})
        fields.update(new)
        if "number" in fields:
            fields = {k: fields[k] for k in ("number", "flight_date", *FARE_FIELDS, "quantity") if k in fields}
        update = {"draft": fields, "pending": intent, "booking_account": None}
        if intent == "status":
            if "number" in fields:
                return {**update, "node": "flight_status"}
            return {**update, "node": "reply", "answer": result("clarify", "Укажите номер рейса.", missing=["number"])}
        missing = [] if "number" in fields else [
            k for k in ("airline", "origin", "destination", "flight_date") if k not in fields
        ]
        if missing:
            return {**update, "node": "reply", "answer": result(
                "clarify", "Укажите номер рейса или авиакомпанию, маршрут и дату вылета.", missing=missing)}
        if state["passengers"] is None:
            return {**update, "node": "check_booking"}
        return {**update, "node": "book_ticket"}

    def status(self, state):
        fields = {k: v for k, v in state["draft"].items() if k in ("number", "flight_date")}
        answer, selected = flight_status(self.store, **fields)
        return {"answer": answer, "selected": selected, "trace": [{"tool": "flight_status", "args": fields}],
                **take_offers(answer),
                **({} if answer["status"] == "clarify" else {"pending": "", "draft": {}})}

    def listing(self, state):
        fields = state["arguments"]
        answer = list_flights(self.store, fields)
        memory = take_offers(answer)
        return {"answer": answer, "pending": "", "draft": {},
                "recent_offers": memory.get("recent_offers", []), "offer_choices": [],
                "trace": [{"tool": "list_flights", "args": fields}]}

    def offer_selection(self, state, *, book):
        ids = set(state["arguments"]["ids"])
        choices = [offer for offer in state.get("recent_offers", []) if offer["offer_id"] in ids]
        quantity = state["arguments"].get("quantity", 1)
        count_fields = {"quantity": quantity} if quantity > 1 else {}
        if book and len(choices) == 1:
            offer = choices[0]
            fields = {key: offer[key] for key in ("number", "flight_date", *FARE_FIELDS)}
            return {"node": "check_booking", "draft": {**fields, **count_fields}, "pending": "book",
                    "booking_account": None, "offer_choices": []}
        flights = {(offer["number"], offer["flight_date"]) for offer in choices}
        if book and quantity > 1 and len(flights) == 1:
            number, flight_date = next(iter(flights))
            return {"node": "check_booking", "draft": {"number": number, "flight_date": flight_date, **count_fields},
                    "pending": "book", "booking_account": None, "offer_choices": choices, "offer_quantity": quantity}
        update = {"node": "reply"}
        if book:
            update.update(pending="", draft={}, booking_account=None, offer_choices=choices, offer_quantity=quantity)
        if not choices:
            answer = result("no_offer", "В последнем списке нет такого варианта. Уточните цену или запросите рейсы ещё раз.")
        elif book:
            answer = result("clarify", "Подошло несколько тарифов. Введите «вариант 1», «вариант 2» и т. д.",
                            missing=["offer"], alternatives=(f'{i}. {o["label"]}' for i, o in enumerate(choices, 1)))
        else:
            answer = result("offer_info", "Варианты из последнего списка.",
                            alternatives=(f'{i}. {o["label"]}' for i, o in enumerate(choices, 1)))
            update.update(recent_offers=[{**o, "flight_position": i} for i, o in enumerate(choices, 1)],
                          offer_choices=[])
        return {**update, "answer": answer}

    def bookings(self, state):
        return {"answer": my_bookings(self.store, state["account_id"]),
                "trace": [{"tool": "my_bookings", "args": {}}]}

    def find_cancellation(self, state):
        fields = {key: value for key, value in state["arguments"].items()
                  if key in {"booking_id", "number", "flight_date"}}
        rows = find_cancellable(self.store, state["account_id"], **fields)
        update = {"trace": [{"tool": "find_cancellable", "args": fields}]}
        if not rows:
            return {**update, "answer": result("no_booking_to_cancel", "Активная бронь не найдена.")}
        if len(rows) > 1:
            answer = result("clarify", "Укажите ID брони, которую хотите отменить.", missing=["booking_id"],
                            alternatives=(f'{row["booking_id"]} · {describe(row)}' for row in rows))
            return {**update, "answer": answer}
        row = rows[0]
        booking_id = str(row["booking_id"])
        message = f'Отменить бронь {booking_id}?\n{describe(row)}\nДля подтверждения введите /cancel и пароль.'
        return {**update, "cancellation": {"account_id": state["account_id"], "booking_id": booking_id},
                "answer": result("cancel_ready", message, missing=["password"])}

    def check(self, state):
        fields = state["draft"]
        row, problem = check_booking(self.store, fields)
        trace = [{"tool": "check_booking", "args": fields}]
        if problem:
            update = {"answer": problem, "trace": trace, **take_offers(problem)}
            if problem["status"] == "no_date":
                update["draft"] = {key: value for key, value in fields.items() if key != "flight_date"}
            elif problem["status"] != "clarify":
                update.update(pending="", draft={}, offer_choices=[])
            return update
        quantity = fields.get("quantity", 1)
        if quantity > 1 and "fare_price" not in fields and state.get("offer_choices"):
            choices = state["offer_choices"]
            return {"trace": trace, "answer": result(
                "clarify", f'Рейс {row["number"]} · билетов: {quantity}. Выберите тариф.',
                missing=["offer"], alternatives=(f'{i}. {offer["label"]}' for i, offer in enumerate(choices, 1)))}
        fare = ("\nВыбран тариф: " + describe_fare(fields["fare_price"], fields["fare_baggage"])
                if "fare_price" in fields else "")
        group = f"\nБилетов: {quantity}." if quantity > 1 else ""
        if quantity > 1 and "fare_price" in fields:
            group += " Итого: " + f'{fields["fare_price"] * quantity:,}'.replace(",", " ") + " ₽."
        passenger = f'Первый пассажир: {state["username"]}.' if quantity > 1 else f'Пассажир: {state["username"]}.'
        answer = result("ready", f'{describe(row)}{fare}{group}\n{passenger} '
                        "Введите /passport. Сначала агент проверит пароль аккаунта.", missing=["password", "passport"])
        return {"answer": answer, "trace": trace, "booking_account": state["account_id"],
                "draft": {"number": row["number"], "flight_date": row["flight_date"].isoformat(),
                          **{key: fields[key] for key in (*FARE_FIELDS, "quantity") if key in fields}}}

    def book(self, state):
        fields = state["draft"]
        answer = book_passengers(self.store, state["account_id"], fields, state["passengers"])
        update = {"answer": answer, **take_offers(answer), "trace": [{"tool": "book_ticket", "args": {
            **fields, "passenger": "профиль", "passport": "[скрыт]"}}]}
        if answer["status"] == "no_date":
            update["draft"] = {key: value for key, value in fields.items() if key != "flight_date"}
        elif answer["status"] == "invalid_passport":
            update["booking_account"] = state["account_id"]
            answer["message"] += " Повторите /passport."
        elif answer["status"] != "clarify":
            update.update(pending="", draft={})
        return update

    def reply(self, state):
        return {"text": "", "passengers": None}


def new_thread():
    return str(uuid4())
