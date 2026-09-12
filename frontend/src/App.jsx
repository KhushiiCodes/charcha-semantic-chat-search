import React, { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  CalendarDays,
  ChevronDown,
  ChevronRight,
  MessageCircle,
  Search,
  Sparkles,
  Users,
} from "lucide-react";

const API_BASE = "http://127.0.0.1:8000";

const FALLBACK_PARTICIPANTS = [
  "Aditya",
  "Aman",
  "Arjun",
  "Khushi",
  "Neha",
  "Priya",
  "Rohan",
  "Sneha",
];

const SUGGESTIONS = [
  "Which destination became the final choice?",
  "Who agreed to go to Manali?",
  "Who owed what for dinner?",
];

function initials(name = "?") {
  return name
    .split(" ")
    .map((part) => part[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

function avatarClass(name = "") {
  const classes = ["pink", "blue", "mint", "purple", "coral", "sky", "violet", "green"];
  const index = [...name].reduce((sum, c) => sum + c.charCodeAt(0), 0) % classes.length;
  return classes[index];
}

function normalizeResult(item, index) {
  const score = Number(item.score ?? item.similarity ?? item.relevance ?? 0);
  const rawContext = item.context ?? item.conversation ?? item.nearby_messages ?? [];
  const context = Array.isArray(rawContext)
    ? rawContext.map((message) => ({
        sender: message.sender ?? message.author ?? "",
        text: message.text ?? message.message ?? message.content ?? String(message),
        id: message.id ?? message.message_id,
      }))
    : [];

  return {
    id: item.id ?? item.message_id ?? `result-${index}`,
    sender: item.sender ?? item.author ?? "Unknown",
    timestamp: item.timestamp ?? item.datetime ?? item.date ?? "",
    text: item.text ?? item.message ?? item.content ?? "",
    score: score > 1 ? score / 100 : score,
    context,
    threadId: item.thread_id ?? item.thread ?? "",
  };
}

function scoreLabel(score) {
  return `${Math.round(Math.max(0, Math.min(1, score)) * 100)}%`;
}

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [participants, setParticipants] = useState(FALLBACK_PARTICIPANTS);
  const [messageCount, setMessageCount] = useState(4200);
  const [dateRange, setDateRange] = useState({ from: "", to: "" });
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const [error, setError] = useState("");
  const [sender, setSender] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [expanded, setExpanded] = useState(null);
  const [replayResult, setReplayResult] = useState(null);
  const inputRef = useRef(null);

  useEffect(() => {
    const onKey = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        inputRef.current?.focus();
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    fetch(`${API_BASE}/stats`)
      .then((response) => {
        if (!response.ok) throw new Error("stats");
        return response.json();
      })
      .then((data) => {
        if (Number.isFinite(data.messages)) setMessageCount(data.messages);
        if (Array.isArray(data.participants) && data.participants.length) {
          setParticipants(data.participants);
        }
        setDateRange({ from: data.date_from ?? "", to: data.date_to ?? "" });
      })
      .catch(() => {});
  }, []);

  const runSearch = async (value = query) => {
    const q = value.trim();
    if (!q) {
      inputRef.current?.focus();
      return;
    }

    setLoading(true);
    setSearched(true);
    setError("");
    setExpanded(null);
    setReplayResult(null);

    try {
      const response = await fetch(`${API_BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          sender: sender || null,
          date_from: startDate || null,
          date_to: endDate || null,
          top_k: 12,
        }),
      });

      if (!response.ok) throw new Error(`Search failed (${response.status})`);

      const data = await response.json();
      const raw = data.results ?? data.matches ?? data.data ?? [];
      setResults(
        Array.isArray(raw)
          ? raw.map((item, index) => ({
              ...normalizeResult(item, index),
              searchQuery: q,
            }))
          : []
      );
    } catch (err) {
      setError(
        "The search service could not be reached. Make sure the FastAPI backend is running on port 8000."
      );
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const clearFilters = () => {
    setSender("");
    setStartDate("");
    setEndDate("");
  };

  const clearSearch = () => {
    setQuery("");
    setResults([]);
    setSearched(false);
    setError("");
    setExpanded(null);
    inputRef.current?.focus();
  };

  const handleSuggestion = (suggestion) => {
    setQuery(suggestion);
    runSearch(suggestion);
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <MessageCircle size={20} strokeWidth={2.4} />
          </div>
          <div>
            <div className="brand-name">Charcha</div>
            <div className="brand-sub">Semantic chat search</div>
          </div>
        </div>

        <div className="topbar-meta">
          <span className="status-dot" />
          <span>{messageCount.toLocaleString()} messages indexed</span>
        </div>
      </header>

      <main className="page">
        <section className="hero">
          <div className="eyebrow">
            <Sparkles size={14} />
            Search by meaning, not just words
          </div>
          <h1>
            Search your <span>conversations.</span>
          </h1>
          <p>
            Find decisions, plans, people and moments hidden inside your group chat.
          </p>

          <div className={`search-shell ${loading ? "searching" : ""}`}>
            <Search size={22} className="search-icon" />
            <input
              ref={inputRef}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && runSearch()}
              placeholder="Ask anything about your conversations..."
              aria-label="Search conversations"
            />
            {query && (
              <button className="clear-search" onClick={clearSearch} aria-label="Clear search">
                ×
              </button>
            )}
            <div className="shortcut">Ctrl K</div>
            <button className="search-submit" onClick={() => runSearch()} disabled={loading}>
              {loading ? <span className="spinner" /> : <>Search <ArrowRight size={17} /></>}
            </button>
          </div>

          <div className="suggestion-row">
            <span>Try:</span>
            {SUGGESTIONS.map((suggestion) => (
              <button key={suggestion} onClick={() => handleSuggestion(suggestion)}>
                {suggestion}
              </button>
            ))}
          </div>
        </section>

        <section className="workspace">
          <div className="results-column">
            <div className="section-header">
              <div>
                <div className="section-kicker">RESULTS</div>
                <div className="results-title">
                  {searched ? "Search results" : "Ready when you are"}
                  {searched && <span>{results.length} found</span>}
                </div>
              </div>
            </div>

            <div className="filter-strip">
              <div className="filter-title">
                <Users size={15} />
                Filter
              </div>

              <label className="filter-control">
                <Users size={14} />
                <select value={sender} onChange={(event) => setSender(event.target.value)}>
                  <option value="">Any person</option>
                  {participants.map((person) => (
                    <option key={person} value={person}>
                      {person}
                    </option>
                  ))}
                </select>
                <ChevronDown size={14} />
              </label>

              <label className="filter-control date-control">
                <CalendarDays size={14} />
                <input
                  type="date"
                  value={startDate}
                  onChange={(event) => setStartDate(event.target.value)}
                  aria-label="From date"
                />
              </label>

              <span className="date-separator">to</span>

              <label className="filter-control date-control">
                <CalendarDays size={14} />
                <input
                  type="date"
                  value={endDate}
                  onChange={(event) => setEndDate(event.target.value)}
                  aria-label="To date"
                />
              </label>

              {(sender || startDate || endDate) && (
                <button className="clear-filter" onClick={clearFilters}>
                  Clear filters
                </button>
              )}
            </div>

            {error && <div className="error-card">{error}</div>}

            {!searched && (
              <div className="welcome-card">
                <div className="welcome-icon">
                  <Search size={22} />
                </div>
                <div>
                  <h2>Ask a natural question</h2>
                  <p>
                    Charcha looks beyond exact keywords, so you can search for an idea even when the
                    original message used completely different words.
                  </p>
                </div>
              </div>
            )}

            {searched && !loading && results.length === 0 && !error && (
              <div className="empty-state">
                <div className="empty-icon">
                  <Search size={23} />
                </div>
                <h3>No matching conversation found</h3>
                <p>Try describing the idea rather than repeating the exact words from the message.</p>
              </div>
            )}

            <div className="results-list">
              {results.map((result, index) => {
                const messageCountInContext = result.context?.length ?? 0;
                const isExpanded = expanded === result.id;

                return (
                  <article className={`result-card ${isExpanded ? "expanded" : ""}`} key={result.id}>
                    <div className="result-rank">{String(index + 1).padStart(2, "0")}</div>
                    <div className={`avatar result-avatar ${avatarClass(result.sender)}`}>
                      {initials(result.sender)}
                    </div>

                    <div className="result-main">
                      <div className="result-meta">
                        <div>
                          <div className="sender-name">{result.sender}</div>
                          <div className="timestamp">{result.timestamp}</div>
                        </div>
                        <div className="match-pill">{scoreLabel(result.score)} match</div>
                      </div>

                      <div className="result-text">{result.text}</div>

                      <button
                        className="replay-button"
                        onClick={() => setReplayResult(result)}
                      >
                        <MessageCircle size={14} />
                        Replay conversation
                        <ChevronRight size={14} />
                      </button>
                    </div>

                    <div className="match-column">
                      <div className="match-track">
                        <div style={{ width: `${Math.max(5, Math.min(100, result.score * 100))}%` }} />
                      </div>
                    </div>

                    <button
                      className="result-arrow"
                      onClick={() => setReplayResult(result)}
                      aria-label="Replay conversation"
                    >
                      <ChevronRight size={18} />
                    </button>
                  </article>
                );
              })}
            </div>
          </div>

          <aside className="info-column">
            <section className="info-card overview-card">
              <div className="card-heading">
                <span className="card-icon">
                  <MessageCircle size={17} />
                </span>
                <div>
                  <h2>Dataset overview</h2>
                  <p>Your searchable conversation space</p>
                </div>
              </div>

              <div className="overview-grid">
                <div>
                  <strong>{messageCount.toLocaleString()}</strong>
                  <span>messages indexed</span>
                </div>
                <div>
                  <strong>{participants.length}</strong>
                  <span>participants</span>
                </div>
                <div>
                  <strong>6 months</strong>
                  <span>of conversations</span>
                </div>
                <div>
                  <strong>Hinglish + English</strong>
                  <span>messaging style</span>
                </div>
              </div>

              {dateRange.from && dateRange.to && (
                <div className="dataset-range">
                  <CalendarDays size={14} />
                  <span>
                    {dateRange.from} <b>→</b> {dateRange.to}
                  </span>
                </div>
              )}
            </section>

            <section className="info-card how-card">
              <div className="how-mark">
                <Sparkles size={17} />
              </div>
              <h2>What Charcha finds</h2>
              <p>
                Ask about a decision, a person, a plan or an event. Semantic matching connects your
                question to messages even when the wording is different.
              </p>
              <div className="how-tags">
                <span>People</span>
                <span>Decisions</span>
                <span>Plans</span>
                <span>Time</span>
              </div>
            </section>

            <section className="participant-card">
              <div className="participant-heading">
                <div>
                  <h2>Participants</h2>
                  <p>Use the filter above to narrow a search.</p>
                </div>
                <span>{participants.length}</span>
              </div>
              <div className="participant-grid">
                {participants.map((person) => (
                  <button
                    key={person}
                    className={sender === person ? "selected" : ""}
                    onClick={() => setSender(sender === person ? "" : person)}
                  >
                    <span className={`avatar tiny ${avatarClass(person)}`}>{initials(person)}</span>
                    {person}
                  </button>
                ))}
              </div>
            </section>
          </aside>
        </section>
      </main>

      {replayResult && (
        <ConversationReplay
          result={replayResult}
          onClose={() => setReplayResult(null)}
        />
      )}

      <footer className="footer">
        <span>Charcha</span>
        <span>•</span>
        <span>Meaning-first search for group conversations</span>
      </footer>
    </div>
  );
}


function ConversationReplay({ result, onClose }) {
  const messages = Array.isArray(result.context) ? result.context : [];

  const allMessages = messages.some((message) => message.id === result.id)
    ? messages
    : [
        ...messages,
        { id: result.id, sender: result.sender, text: result.text },
      ];

  const isDecisionQuery = /decide|decision|final|choose|choice|agreed|agree|plan|fixed|fix/i.test(
    result.searchQuery || ""
  );

  return (
    <div className="replay-overlay" onClick={onClose}>
      <div
        className="replay-modal"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="replay-header">
          <div>
            <div className="replay-kicker">
              <MessageCircle size={14} />
              CONVERSATION REPLAY
            </div>
            <h2>Follow the conversation</h2>
            <p>The surrounding messages that led to this result.</p>
          </div>

          <button
            className="replay-close"
            onClick={onClose}
            aria-label="Close conversation"
          >
            ×
          </button>
        </div>

        <div className="replay-meta">
          <span>{result.sender}</span>
          <span>•</span>
          <span>{result.timestamp}</span>
        </div>

        <div className="replay-thread">
          {allMessages.map((message, index) => {
            const isResult =
              message.id === result.id ||
              (message.sender === result.sender && message.text === result.text);

            return (
              <React.Fragment key={`${message.id || index}-${index}`}>
                <div className={`replay-message ${isResult ? "replay-message-result" : ""}`}>
                  <div className={`avatar replay-avatar ${avatarClass(message.sender)}`}>
                    {initials(message.sender)}
                  </div>

                  <div className="replay-message-body">
                    <div className="replay-sender">
                      {message.sender}
                      {isResult && <span className="result-badge">SEARCH MATCH</span>}
                    </div>
                    <div className="replay-text">{message.text}</div>
                  </div>
                </div>

                {index < allMessages.length - 1 && (
                  <div className="replay-connector"><span /></div>
                )}

                {isResult && isDecisionQuery && (
                  <div className="decision-card">
                    <div className="decision-icon">✓</div>
                    <div>
                      <strong>Decision point</strong>
                      <p>
                        This message appears to be the point where the conversation reached its answer.
                      </p>
                    </div>
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>

        <div className="replay-footer">
          <div>
            <strong>{allMessages.length}</strong>
            <span>messages in this conversation</span>
          </div>
          <button onClick={onClose}>
            Back to results
            <ArrowRight size={15} />
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
