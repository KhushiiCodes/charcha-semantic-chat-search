import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from embeddings import Embedder


BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "messages.json"
CACHE = Path(__file__).resolve().parent / ".charcha_embeddings.npz"


# =========================================================
# CONSTANTS
# =========================================================

STOPWORDS = {
    "what", "did", "does", "do", "we", "i", "you", "he", "she", "they",
    "the", "a", "an", "about", "say", "said", "tell", "me", "which",
    "when", "where", "was", "were", "is", "are", "our", "their", "this",
    "that", "it", "for", "of", "to", "in", "on", "and", "or", "with",
    "decide", "decision", "final", "choice", "became", "become",
    "last", "month", "week", "day", "time", "conversation", "chat",
    "can", "could", "would", "please", "show", "find", "who", "how",
    "did", "go", "was", "were", "we",
}


DECISION_TERMS = {
    "decide", "decided", "decision", "final", "choice", "chosen", "choose",
    "fix", "fixed", "settled", "confirm", "confirmed", "confirmation",
    "book", "booked", "plan", "planned", "select", "selected", "agree",
    "agreed", "lock", "locked", "done", "finalized", "finalise",
    "finalised", "settle", "accepted", "accept", "support",
}


AGREEMENT_TERMS = {
    "yes", "haan", "ha", "okay", "ok", "done", "doneee",
    "agreed", "agree", "confirmed", "confirm", "accepted",
    "accept", "sure", "works", "fine", "in", "main", "chalo",
    "let", "lets", "kar", "karta", "karti",
}


DESTINATION_TERMS = {
    "manali", "kasol", "shimla",
    "destination", "trip", "travel", "vacation", "holiday",
    "place", "location", "go", "going",
}


ACTION_TERMS = {
    "book", "booked", "booking",
    "buy", "bought",
    "fill", "filled",
    "submit", "submitted",
    "send", "sent",
    "pay", "paid",
    "choose", "chosen",
    "select", "selected",
    "confirm", "confirmed",
    "fix", "fixed",
    "lock", "locked",
}


SHORT_NOISE = {
    "ok", "okay", "haan", "yes", "no", "ya", "yep", "yup",
    "bruh", "lol", "hmm", "hm", "wait", "fix", "?",
    "achha", "acha", "good", "nice", "cool", "same", "true",
    "exactly", "forwarded", "sent", "dekhta", "dekhta hu",
}


TIME_PATTERNS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text):
    return {
        t
        for t in normalize(text).split()
        if len(t) > 1 and t not in STOPWORDS
    }


# =========================================================
# SEARCH ENGINE
# =========================================================

class ChatSearch:

    def __init__(self):

        # -------------------------------------------------
        # Load messages
        # -------------------------------------------------

        with open(DATA, encoding="utf-8") as f:
            self.messages = json.load(f)

        self.by_id = {
            m["id"]: i
            for i, m in enumerate(self.messages)
        }

        self.senders = sorted({
            m["sender"]
            for m in self.messages
        })

        # -------------------------------------------------
        # Group messages by thread
        # -------------------------------------------------

        self.thread_messages = defaultdict(list)

        for message in self.messages:
            thread_id = message.get(
                "thread_id",
                "default"
            )
            self.thread_messages[thread_id].append(message)

        for thread_id in self.thread_messages:
            self.thread_messages[thread_id].sort(
                key=lambda x: x["timestamp"]
            )

        # Position of every message in its thread
        self.thread_position = {}

        for thread_id, messages in self.thread_messages.items():
            for position, message in enumerate(messages):
                self.thread_position[message["id"]] = position

        # -------------------------------------------------
        # Embed message + local conversation context
        # -------------------------------------------------

        self.embedder = Embedder()

        self.embeddings = self._load_or_build_embeddings()

        # -------------------------------------------------
        # Dataset dates
        # -------------------------------------------------

        self.dataset_min = min(
            m["timestamp"][:10]
            for m in self.messages
        )

        self.dataset_max = max(
            m["timestamp"][:10]
            for m in self.messages
        )

        # -------------------------------------------------
        # Thread embeddings
        # -------------------------------------------------

        self.thread_embedding = {}

        for thread_id, messages in self.thread_messages.items():

            vectors = []

            for message in messages:
                index = self.by_id[message["id"]]
                vectors.append(self.embeddings[index])

            if vectors:

                matrix = np.asarray(vectors)

                vector = matrix.mean(axis=0)

                norm = np.linalg.norm(vector)

                if norm > 0:
                    vector = vector / norm

                self.thread_embedding[thread_id] = (
                    vector.astype("float32")
                )

    # =====================================================
    # CONTEXT
    # =====================================================

    def _context_text(self, message, radius=3):

        thread_id = message.get(
            "thread_id",
            "default"
        )

        messages = self.thread_messages[thread_id]

        position = self.thread_position[
            message["id"]
        ]

        lo = max(
            0,
            position - radius
        )

        hi = min(
            len(messages),
            position + radius + 1
        )

        parts = []

        for item in messages[lo:hi]:

            parts.append(
                f'{item["sender"]}: {item["text"]}'
            )

        return " | ".join(parts)

    # =====================================================
    # EMBEDDINGS
    # =====================================================

    def _load_or_build_embeddings(self):

        if CACHE.exists():

            try:

                cached = np.load(
                    CACHE,
                    allow_pickle=False
                )

                arr = cached["embeddings"]

                count = int(cached["count"])

                if (
                    count == len(self.messages)
                    and arr.shape[0] == len(self.messages)
                ):
                    return arr.astype("float32")

            except Exception:
                pass

        texts = [
            self._context_text(message)
            for message in self.messages
        ]

        arr = self.embedder.encode(texts)

        try:

            np.savez_compressed(
                CACHE,
                embeddings=arr,
                count=np.array(len(self.messages))
            )

        except Exception:
            pass

        return arr

    # =====================================================
    # SENDER DETECTION
    # =====================================================

    def _detect_sender(
        self,
        query,
        explicit_sender
    ):

        if explicit_sender:
            return explicit_sender

        query_lower = query.lower()

        for sender in sorted(
            self.senders,
            key=len,
            reverse=True
        ):

            if re.search(
                r"\b" + re.escape(sender.lower()) + r"\b",
                query_lower
            ):
                return sender

        return None

    # =====================================================
    # DATE PARSING
    # =====================================================

    def _parse_date_range(
        self,
        query,
        date_from=None,
        date_to=None
    ):

        if date_from or date_to:
            return date_from, date_to

        q = query.lower()

        latest = datetime.fromisoformat(
            self.dataset_max
        )

        if "last week" in q:

            end = latest

            start = end - timedelta(days=7)

            return (
                start.date().isoformat(),
                end.date().isoformat()
            )

        if "this week" in q:

            end = latest

            start = end - timedelta(days=6)

            return (
                start.date().isoformat(),
                end.date().isoformat()
            )

        if "last month" in q:

            first_current = latest.replace(day=1)

            previous_last = (
                first_current - timedelta(days=1)
            )

            previous_first = (
                previous_last.replace(day=1)
            )

            return (
                previous_first.date().isoformat(),
                previous_last.date().isoformat()
            )

        if "this month" in q:

            first = latest.replace(day=1)

            return (
                first.date().isoformat(),
                latest.date().isoformat()
            )

        for name, month in TIME_PATTERNS.items():

            if re.search(
                r"\b" + re.escape(name) + r"\b",
                q
            ):

                year_match = re.search(
                    r"\b(20\d{2})\b",
                    q
                )

                year = (
                    int(year_match.group(1))
                    if year_match
                    else latest.year
                )

                start = datetime(
                    year,
                    month,
                    1
                )

                if month == 12:

                    end = (
                        datetime(year + 1, 1, 1)
                        - timedelta(days=1)
                    )

                else:

                    end = (
                        datetime(year, month + 1, 1)
                        - timedelta(days=1)
                    )

                return (
                    start.date().isoformat(),
                    end.date().isoformat()
                )

        match = re.search(
            r"\b(20\d{2}-\d{2}-\d{2})\b",
            q
        )

        if match:

            value = match.group(1)

            return value, value

        return None, None

    # =====================================================
    # CONTENT QUERY
    # =====================================================

    def _content_query(
        self,
        query,
        sender
    ):

        q = query

        if sender:

            q = re.sub(
                r"\b" + re.escape(sender) + r"\b",
                " ",
                q,
                flags=re.IGNORECASE
            )

        patterns = [

            r"\bwhat did\b",
            r"\bwhat does\b",
            r"\bwhat was\b",
            r"\bwhat were\b",
            r"\bwhat is\b",
            r"\bwhat are\b",

            r"\bwhich\b",
            r"\bwhen did\b",
            r"\bwhen was\b",
            r"\bwhere did\b",

            r"\bwho did\b",
            r"\bwho was\b",
            r"\bwho were\b",
            r"\bwho\b",

            r"\btell me\b",
            r"\bcan you tell me\b",
            r"\bshow me\b",

            r"\bwhat\b",
            r"\babout\b",
            r"\bsay\b",
            r"\bsaid\b",

            r"\bfinal choice\b",
            r"\bfinal decision\b",
            r"\bfinal destination\b",

            r"\bbecame the\b",
            r"\bwas the\b",

            # WHO/AGREEMENT wording
            r"\bagreed to\b",
            r"\bagreed on\b",
            r"\bagree to\b",
            r"\bagree on\b",
            r"\bconfirmed the\b",
            r"\bconfirmed to\b",
            r"\bwho was in\b",
            r"\bwho accepted\b",
            r"\bwho supported\b",
        ]

        for pattern in patterns:

            q = re.sub(
                pattern,
                " ",
                q,
                flags=re.IGNORECASE
            )

        q = re.sub(
            r"\s+",
            " ",
            q
        ).strip()

        return q

    # =====================================================
    # QUERY INTENT
    # =====================================================

    def _query_intent(
        self,
        query,
        content_query
    ):

        q = query.lower()

        query_tokens = tokens(
            content_query or query
        )

        # -----------------------------------------------
        # Decision intent
        # -----------------------------------------------

        decision = bool(
            query_tokens & DECISION_TERMS
            or any(
                phrase in q
                for phrase in [
                    "decide",
                    "decision",
                    "final",
                    "choice",
                    "chosen",
                    "fix",
                    "settle",
                    "settled",
                    "confirm",
                    "confirmed",
                    "locked",
                    "agreed",
                    "agree",
                ]
            )
        )

        # -----------------------------------------------
        # Destination intent
        # -----------------------------------------------

        destination = bool(
            query_tokens & DESTINATION_TERMS
            or "destination" in q
            or "trip" in q
            or "where are we going" in q
            or "where did we go" in q
            or "go to" in q
        )

        # -----------------------------------------------
        # Person/quote intent
        # -----------------------------------------------

        person = bool(
            "what did" in q
            or "what does" in q
            or "said" in q
            or "say" in q
        )

        # -----------------------------------------------
        # WHO AGREEMENT intent
        # -----------------------------------------------

        who_agreement = bool(
            re.search(
                r"\bwho\b.*\b(agreed|agree|confirmed|accepted|"
                r"supported|in|joined)\b",
                q
            )
            or re.search(
                r"\bwho\b.*\bfinal plan\b",
                q
            )
            or re.search(
                r"\bwho\b.*\bdecision\b",
                q
            )
        )

        # -----------------------------------------------
        # Time intent
        # -----------------------------------------------

        time = bool(
            "when" in q
            or "last month" in q
            or "this month" in q
            or "last week" in q
            or "this week" in q
            or bool(
                re.search(
                    r"\b20\d{2}-\d{2}-\d{2}\b",
                    q
                )
            )
        )

        return {
            "decision": decision,
            "destination": destination,
            "person": person,
            "who_agreement": who_agreement,
            "time": time,
            "tokens": query_tokens,
        }

    # =====================================================
    # SCORE HELPERS
    # =====================================================

    def _lexical_score(
        self,
        query,
        text
    ):

        query_tokens = tokens(query)

        text_tokens = tokens(text)

        if not query_tokens or not text_tokens:
            return 0.0

        overlap = (
            query_tokens &
            text_tokens
        )

        return len(overlap) / max(
            1,
            len(query_tokens)
        )

    def _decision_score(self, text):

        text_tokens = set(
            normalize(text).split()
        )

        if not text_tokens:
            return 0.0

        hits = len(
            text_tokens &
            DECISION_TERMS
        )

        return min(
            1.0,
            hits / 2.0
        )

    def _destination_score(self, text):

        text_tokens = set(
            normalize(text).split()
        )

        hits = len(
            text_tokens &
            DESTINATION_TERMS
        )

        return min(
            1.0,
            hits / 2.0
        )

    def _action_score(self, text):

        text_tokens = set(
            normalize(text).split()
        )

        hits = len(
            text_tokens &
            ACTION_TERMS
        )

        return min(
            1.0,
            hits / 2.0
        )

    # =====================================================
    # MESSAGE QUALITY
    # =====================================================

    def _message_quality(self, text):

        normalized = normalize(text)

        words = normalized.split()

        if not words:
            return 0.0

        if len(words) == 1:

            word = words[0]

            if word in SHORT_NOISE:
                return 0.0

            if word in DECISION_TERMS:
                return 0.45

            if word in DESTINATION_TERMS:
                return 0.65

            return 0.35

        if len(words) == 2:
            return 0.35

        if len(words) == 3:
            return 0.55

        return 0.75

    # =====================================================
    # AGREEMENT SCORE
    #
    # NEW:
    # Specifically identifies messages that represent
    # a person agreeing/confirming/joining a decision.
    # =====================================================

    def _agreement_score(self, text):

        normalized = normalize(text)

        words = normalized.split()

        if not words:
            return 0.0

        score = 0.0

        # -----------------------------------------------
        # Explicit agreement
        # -----------------------------------------------

        if any(
            word in words
            for word in [
                "agreed",
                "agree",
                "confirmed",
                "confirm",
                "accepted",
                "accept",
                "works",
                "sure",
            ]
        ):
            score += 0.80

        # -----------------------------------------------
        # "done" style confirmation
        # -----------------------------------------------

        if normalized in {
            "done",
            "doneee",
            "okay",
            "ok",
            "haan",
            "yes",
            "sure",
            "fine",
        }:
            score += 0.55

        # -----------------------------------------------
        # "main bhi in" / "I'm in" type messages
        # -----------------------------------------------

        if (
            "in" in words
            and (
                "main" in words
                or "im" in words
                or "i" in words
            )
        ):
            score += 1.0

        # -----------------------------------------------
        # "600 each works" type confirmation
        # -----------------------------------------------

        if "works" in words:
            score += 0.70

        # -----------------------------------------------
        # Avoid treating random "done" as strong unless
        # the surrounding thread is relevant.
        # The thread-level score handles this.
        # -----------------------------------------------

        return min(
            1.0,
            score
        )

    # =====================================================
    # THREAD AGREEMENT SCORE
    #
    # Looks at the complete chronological thread.
    # =====================================================

    def _thread_agreement_score(
        self,
        message,
        thread_messages
    ):

        position = self.thread_position[
            message["id"]
        ]

        message_score = self._agreement_score(
            message["text"]
        )

        # Look around the message for an actual decision.
        lo = max(
            0,
            position - 4
        )

        hi = min(
            len(thread_messages),
            position + 5
        )

        nearby = thread_messages[lo:hi]

        decision_score = 0.0
        destination_score = 0.0

        for item in nearby:

            decision_score = max(
                decision_score,
                self._decision_score(
                    item["text"]
                )
            )

            destination_score = max(
                destination_score,
                self._destination_score(
                    item["text"]
                )
            )

        # Agreement is much more meaningful when it occurs
        # inside a thread that actually contains a decision.
        result = (
            0.55 * message_score
            + 0.25 * decision_score
            + 0.20 * destination_score
        )

        return min(
            1.0,
            result
        )

    # =====================================================
    # ANCHOR SCORE
    # =====================================================

    def _anchor_score(
        self,
        message,
        thread_text,
        intent
    ):

        text = message["text"]

        decision = self._decision_score(text)

        destination = self._destination_score(text)

        action = self._action_score(text)

        quality = self._message_quality(text)

        score = 0.0

        if intent["decision"]:

            score += 0.35 * decision
            score += 0.20 * action

        if intent["destination"]:

            score += 0.35 * destination
            score += 0.15 * decision

        score += 0.20 * quality

        normalized = normalize(text)

        message_tokens = set(
            normalized.split()
        )

        has_destination = bool(
            message_tokens &
            DESTINATION_TERMS
        )

        has_decision = bool(
            message_tokens &
            DECISION_TERMS
        )

        if (
            intent["decision"]
            and intent["destination"]
            and has_destination
            and has_decision
            and len(message_tokens) >= 3
        ):
            score += 0.45

        return min(
            1.0,
            score
        )

    # =====================================================
    # SEARCH
    # =====================================================

    def search(
        self,
        query,
        sender=None,
        date_from=None,
        date_to=None,
        top_k=8
    ):

        query = query.strip()

        if not query:
            return []

        # -------------------------------------------------
        # Understand query
        # -------------------------------------------------

        sender = self._detect_sender(
            query,
            sender
        )

        date_from, date_to = (
            self._parse_date_range(
                query,
                date_from,
                date_to
            )
        )

        content_query = self._content_query(
            query,
            sender
        )

        intent = self._query_intent(
            query,
            content_query
        )

        # -------------------------------------------------
        # Query embeddings
        # -------------------------------------------------

        q_vectors = self.embedder.encode(
            [
                query,
                content_query or query
            ]
        )

        full_query_vector = q_vectors[0]

        topic_query_vector = q_vectors[1]

        dense_full = (
            self.embeddings
            @ full_query_vector
        )

        dense_topic = (
            self.embeddings
            @ topic_query_vector
        )

        dense_message = np.maximum(
            dense_full,
            dense_topic
        )

        # -------------------------------------------------
        # Thread semantic scores
        # -------------------------------------------------

        thread_scores = {}

        for thread_id, vector in (
            self.thread_embedding.items()
        ):

            full_score = float(
                np.dot(
                    full_query_vector,
                    vector
                )
            )

            topic_score = float(
                np.dot(
                    topic_query_vector,
                    vector
                )
            )

            thread_scores[thread_id] = max(
                full_score,
                topic_score
            )

        # -------------------------------------------------
        # Score messages
        # -------------------------------------------------

        scored = []

        for i, message in enumerate(
            self.messages
        ):

            # ---------------------------------------------
            # Sender filter
            # ---------------------------------------------

            if sender:

                if (
                    message["sender"].lower()
                    != sender.lower()
                ):
                    continue

            # ---------------------------------------------
            # Date filters
            # ---------------------------------------------

            message_date = (
                message["timestamp"][:10]
            )

            if (
                date_from
                and message_date < date_from
            ):
                continue

            if (
                date_to
                and message_date > date_to
            ):
                continue

            thread_id = message.get(
                "thread_id",
                "default"
            )

            thread = self.thread_messages[
                thread_id
            ]

            # ---------------------------------------------
            # Local context
            # ---------------------------------------------

            local_context = self._context_text(
                message,
                radius=3
            )

            # ---------------------------------------------
            # Semantic
            # ---------------------------------------------

            semantic = float(
                dense_message[i]
            )

            thread_semantic = float(
                thread_scores.get(
                    thread_id,
                    0.0
                )
            )

            # ---------------------------------------------
            # Lexical
            # ---------------------------------------------

            lexical = self._lexical_score(
                content_query or query,
                local_context
            )

            # ---------------------------------------------
            # Decision
            # ---------------------------------------------

            decision = 0.0

            if intent["decision"]:

                decision = max(
                    self._decision_score(
                        local_context
                    ),
                    self._decision_score(
                        message["text"]
                    )
                )

            # ---------------------------------------------
            # Destination
            # ---------------------------------------------

            destination = 0.0

            if intent["destination"]:

                destination = max(
                    self._destination_score(
                        local_context
                    ),
                    self._destination_score(
                        message["text"]
                    )
                )

            # ---------------------------------------------
            # Anchor
            # ---------------------------------------------

            anchor = self._anchor_score(
                message,
                local_context,
                intent
            )

            # ---------------------------------------------
            # Quality
            # ---------------------------------------------

            quality = self._message_quality(
                message["text"]
            )

            # ---------------------------------------------
            # Date
            # ---------------------------------------------

            date_score = (
                1.0
                if (
                    date_from
                    or date_to
                )
                else 0.0
            )

            # ---------------------------------------------
            # NEW: agreement score
            # ---------------------------------------------

            agreement = 0.0

            if intent["who_agreement"]:

                agreement = (
                    self._thread_agreement_score(
                        message,
                        thread
                    )
                )

            # ---------------------------------------------
            # Noise penalty
            # ---------------------------------------------

            normalized_message = normalize(
                message["text"]
            )

            word_count = len(
                normalized_message.split()
            )

            noise_penalty = 0.0

            if normalized_message in SHORT_NOISE:
                noise_penalty = 0.35

            elif word_count == 1:
                noise_penalty = 0.15

            # ---------------------------------------------
            # FINAL SCORE
            # ---------------------------------------------

            final_score = (
                0.30 * semantic
                + 0.20 * thread_semantic
                + 0.10 * lexical
                + 0.10 * decision
                + 0.08 * destination
                + 0.07 * anchor
                + 0.05 * quality
                + 0.05 * date_score
                + 0.15 * agreement
                - noise_penalty
            )

            # ---------------------------------------------
            # Special WHO AGREEMENT ranking
            #
            # For a "who agreed..." question:
            #
            # - relevant thread matters strongly
            # - agreement message matters strongly
            # - random "done" elsewhere gets penalized
            # ---------------------------------------------

            if intent["who_agreement"]:

                # Strong boost for an actual agreement.
                if agreement >= 0.65:
                    final_score += 0.25

                # If the message itself looks like an
                # agreement, give it another small boost.
                message_agreement = (
                    self._agreement_score(
                        message["text"]
                    )
                )

                if message_agreement >= 0.65:
                    final_score += 0.20

                # If the thread contains the requested
                # destination/topic, strengthen it.
                if destination > 0:
                    final_score += 0.12

                # Generic "done" outside a relevant thread
                # should not dominate the search.
                if (
                    normalized_message in {
                        "done",
                        "doneee",
                        "okay",
                        "ok",
                        "haan",
                    }
                    and destination == 0
                    and decision == 0
                ):
                    final_score -= 0.20

            # ---------------------------------------------
            # Strong explicit decision anchor
            # ---------------------------------------------

            if (
                intent["decision"]
                and intent["destination"]
            ):

                message_tokens = set(
                    normalized_message.split()
                )

                has_destination = bool(
                    message_tokens &
                    DESTINATION_TERMS
                )

                has_decision = bool(
                    message_tokens &
                    DECISION_TERMS
                )

                if (
                    has_destination
                    and has_decision
                    and word_count >= 3
                ):
                    final_score += 0.25

            scored.append(
                (
                    final_score,
                    i,
                    semantic,
                    thread_semantic,
                    lexical,
                    decision,
                    destination,
                    anchor,
                    quality,
                    agreement,
                )
            )

        # -------------------------------------------------
        # Sort
        # -------------------------------------------------

        scored.sort(
            key=lambda x: x[0],
            reverse=True
        )

        # -------------------------------------------------
        # Build results
        # -------------------------------------------------

        results = []

        for (
            final_score,
            i,
            semantic,
            thread_semantic,
            lexical,
            decision,
            destination,
            anchor,
            quality,
            agreement,
        ) in scored[:top_k]:

            message = self.messages[i]

            thread_id = message.get(
                "thread_id",
                "default"
            )

            thread = self.thread_messages[
                thread_id
            ]

            position = self.thread_position[
                message["id"]
            ]

            lo = max(
                0,
                position - 3
            )

            hi = min(
                len(thread),
                position + 4
            )

            result = {
                **message,

                "score": round(
                    float(final_score),
                    4
                ),

                "relevance": round(
                    max(
                        0.0,
                        min(
                            100.0,
                            float(final_score) * 100
                        )
                    ),
                    1
                ),

                "score_breakdown": {

                    "semantic": round(
                        semantic,
                        4
                    ),

                    "thread_semantic": round(
                        thread_semantic,
                        4
                    ),

                    "lexical": round(
                        lexical,
                        4
                    ),

                    "decision": round(
                        decision,
                        4
                    ),

                    "destination": round(
                        destination,
                        4
                    ),

                    "anchor": round(
                        anchor,
                        4
                    ),

                    "quality": round(
                        quality,
                        4
                    ),

                    "agreement": round(
                        agreement,
                        4
                    ),
                },

                "context": thread[lo:hi],
            }

            results.append(result)

        return results