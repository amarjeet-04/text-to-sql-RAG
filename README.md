# 🤖 Conversational Text-to-SQL RAG Chatbot

A **natural, human-like chatbot** that lets you explore your data through conversation. Not just a query tool - it remembers context, asks clarifying questions, and responds like a helpful colleague.

---

## ✨ What Makes This Different?

### ❌ Typical Chatbot
```
User: Show top agents
Bot:  Found 5 results.

User: Filter by India
Bot:  Error: No context. Please provide full query.
```

### ✅ This Chatbot
```
User: Show top agents
Bot:  Here are the top 5 agents! 🎉
      1. Alpha Travel | AED 125,000
      2. Beta Tours | AED 98,000
      ...
      💡 Want to see a specific region?

User: Filter by India
Bot:  Narrowing down to India region...
      
      Here's what I found (3 results):
      1. Mumbai Tours | AED 45,000
      ...
      💡 Should I break this down by month?

User: What about last month?
Bot:  Looking at last month now...
      [Shows India agents for last month, keeping the filter!]
```

---

## 🎯 Key Features

### 1. 💬 Natural Conversations
- **Friendly responses** - Not robotic "Found X results"
- **Personality** - Occasional emojis, encouragement
- **Mood awareness** - Adjusts tone if user seems frustrated

### 2. 🧠 Session Memory  
- **Context tracking** - Remembers tables, filters, time ranges
- **Follow-up detection** - "Filter by X" adds to previous query
- **Smart replacement** - "Show Y instead" swaps filters

### 3. ❓ Clarifying Questions
- **Ambiguous input** - "Did you mean X or Y?"
- **Incomplete queries** - "Could you tell me more about..."
- **Multiple options** - Presents choices when unclear

### 4. 💡 Smart Suggestions
- **Context-aware** - Based on current conversation
- **Actionable** - One-click follow-ups
- **Helpful** - Guide users to insights

### 5. 🎭 Intent Detection
Automatically detects:
- **Greetings** - "Hi" → Friendly welcome
- **Thanks** - "Thank you" → Warm response
- **Help** - "What can you do?" → Capabilities
- **Reset** - "Start over" → Fresh context
- **Complaints** - "That's wrong" → Apologize + retry

---

## 📁 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONVERSATIONAL RAG CHATBOT                       │
│                                                                     │
│  User Message                                                       │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐                                                   │
│  │   Intent    │──▶ Greeting? Thanks? Help? Query?                 │
│  │  Detector   │                                                   │
│  └─────────────┘                                                   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│  │Conversation │───▶│    RAG      │───▶│  Text-to-   │            │
│  │   Memory    │    │  Retrieval  │    │    SQL      │            │
│  │  (Context)  │    │  (Schema)   │    │             │            │
│  └─────────────┘    └─────────────┘    └─────────────┘            │
│       │                                       │                     │
│       │                                       ▼                     │
│       │                               ┌─────────────┐              │
│       │                               │  Response   │              │
│       │◀──── Update Context ──────────│  Generator  │              │
│       │                               │(Personality)│              │
│       │                               └─────────────┘              │
│       │                                       │                     │
│       ▼                                       ▼                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    Natural Response                          │  │
│  │  "Here are the top 5 agents! Alpha Travel leads with..."    │  │
│  │  💡 Want to filter by region?                                │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
text-to-sql-rag/
├── app/
│   ├── main.py                      # FastAPI + Chat UI
│   ├── memory/
│   │   └── conversation_memory.py   # Session & intent tracking
│   ├── rag/
│   │   └── rag_engine.py            # Schema retrieval
│   ├── conversation/
│   │   └── response_generator.py    # Natural responses
│   └── services/
│       ├── chat_service.py          # Main orchestrator
│       └── database.py              # SQL execution
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Configure
```bash
export OPENAI_API_KEY=sk-your-key
export LLM_PROVIDER=openai   # or 'ollama'
export LLM_MODEL=gpt-4       # or 'llama3'
```

### 3. Run
```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Chat!
Open http://localhost:8000/chat

---

## 💬 Conversation Examples

### Example 1: Building a Query
```
You: Hi!
Bot: Hi there! 👋 I'm your data assistant. Ask me anything about 
     bookings, agents, or revenue!

You: Show top agents
Bot: Here are the top 5 agents! 
     1. Alpha Travel | AED 125,000
     2. Beta Tours | AED 98,000
     ...
     💡 Want to see a specific region?

You: Just from India
Bot: Narrowing down to India region...
     
     Here's what I found (3 results):
     1. Mumbai Tours | AED 45,000
     ...

You: What about last month?
Bot: Looking at last month now...
     [Keeps India filter + adds time filter]

You: Show Europe instead
Bot: Switching to Europe instead of India...
     [Replaces India with Europe, keeps last month]
```

### Example 2: Clarification
```
You: Show revenue
Bot: I can show you a few different things here:
     • Total revenue (sum of booking values)
     • Revenue by agent
     • Revenue by nationality
     • Revenue trend over time
     
     Which would be most helpful?

You: By nationality
Bot: Here's revenue breakdown by nationality this month:
     ...
```

### Example 3: Error Recovery
```
You: asdfghjk
Bot: Hmm, I didn't quite catch that. 🤔 
     Could you try rephrasing? 
     
     For example:
     • "Show top agents by revenue"
     • "How many bookings today?"

You: top agents
Bot: Got it! Here are the top 5 agents...
```

---

## 🎨 Response Personality

### Greetings
```
"Hi there! 👋 I'm your data assistant..."
"Hello! Ready to help you explore your data..."
"Hey! 👋 I can help you query your booking data..."
```

### Success
```
"Here's what I found:"
"Got it! Here's your answer:"
"Found {count} results for you:"
```

### Errors (Sympathetic)
```
"Hmm, I ran into a snag with that query..."
"Oops! Something didn't work quite right..."
```

### Errors (Encouraging - after multiple failures)
```
"No worries, let's try again! Maybe something like..."
"That's okay! Here are some things I know I can help with..."
```

### Follow-up Transitions
```
"Got it! Filtering by India..."
"Sure! Switching to Europe instead..."
"Looking at last month now..."
```

---

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key |
| `LLM_PROVIDER` | openai | Provider: openai, ollama |
| `LLM_MODEL` | gpt-4 | Model name |
| `DB_SERVER` | - | SQL Server host |
| `DB_NAME` | - | Database name |
| `DB_USER` | - | Database user |
| `DB_PASSWORD` | - | Database password |

---

## 🎭 Intent Detection

| User Says | Detected Intent | Bot Action |
|-----------|-----------------|------------|
| "Hi" / "Hello" | `GREETING` | Friendly welcome |
| "Thanks" / "Thank you" | `THANKS` | Warm response |
| "Help" / "What can you do" | `HELP` | Show capabilities |
| "Start over" / "Reset" | `RESET` | Clear context |
| "That's wrong" | `COMPLAINT` | Apologize + retry |
| "Filter by X" | `FOLLOWUP_FILTER` | Add filter |
| "Show Y instead" | `FOLLOWUP_CHANGE` | Replace filter |
| "More details" | `FOLLOWUP_MORE` | Expand results |
| [Data question] | `DATA_QUERY` | Generate SQL |

---

## 🔄 Context Flow

```
Message 1: "Show top agents"
┌─────────────────────────────────────┐
│ Context: {                          │
│   tables: ['Agents', 'Bookings'],   │
│   filters: {},                      │
│   time_range: null                  │
│ }                                   │
└─────────────────────────────────────┘

Message 2: "Filter by India"
┌─────────────────────────────────────┐
│ Context: {                          │
│   tables: ['Agents', 'Bookings'],   │
│   filters: {Region: 'India'},  ← +  │
│   time_range: null                  │
│ }                                   │
└─────────────────────────────────────┘

Message 3: "What about last month?"
┌─────────────────────────────────────┐
│ Context: {                          │
│   tables: ['Agents', 'Bookings'],   │
│   filters: {Region: 'India'},       │
│   time_range: 'last_month'     ← +  │
│ }                                   │
└─────────────────────────────────────┘

Message 4: "Show Europe instead"
┌─────────────────────────────────────┐
│ Context: {                          │
│   tables: ['Agents', 'Bookings'],   │
│   filters: {Region: 'Europe'}, ← ⟳  │
│   time_range: 'last_month'          │
│ }                                   │
└─────────────────────────────────────┘
```

---

## 📊 API Endpoints

### Chat
```bash
POST /api/chat
{
  "message": "Show top agents",
  "session_id": null  # First message
}

# Response
{
  "response": "Here are the top 5 agents!...",
  "session_id": "chat_abc123",
  "sql": "SELECT TOP 5...",
  "data": [...],
  "suggestions": ["Filter by region?", ...],
  "intent": "data_query",
  "context": {...}
}
```

### History
```bash
GET /api/history/{session_id}
```

### Clear Context
```bash
POST /api/clear/{session_id}
```

---

## 🎯 Best Practices

### For Users
1. **Start simple** - "Show top agents"
2. **Build up** - "Filter by India"
3. **Explore** - "What about last month?"
4. **Reset when needed** - "Start over"

### For Developers
1. **Customize responses** in `response_generator.py`
2. **Add schema** in `rag_engine.py`
3. **Tune intents** in `conversation_memory.py`
4. **Adjust personality** for your brand

---

## 📄 License

MIT
