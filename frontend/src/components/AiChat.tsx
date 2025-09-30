import React, { useState } from "react";
import { CohereClient } from "cohere-ai";

interface AiChatProps {
  videoId?: string;
}

const cohereToken = import.meta.env.VITE_COHERE_API_KEY as string | undefined;
const co = cohereToken ? new CohereClient({ token: cohereToken }) : undefined;

const AiChat: React.FC<AiChatProps> = ({ videoId }) => {
  const [messages, setMessages] = useState<
    { sender: "user" | "ai"; text: string }[]
  >([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  // Keep prop wired without affecting behavior (answers are general-purpose)
  if (videoId) {
    // no-op: reserved for future context-aware enhancements
  }

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Add user message
    setMessages((prev) => [...prev, { sender: "user", text: input }]);
    setLoading(true);

    try {
      // Prefer backend RAG when available
      try {
        // Irrespective of video context, always ask a general question
        const payload = { question: input };

        const resp = await fetch('http://localhost:5000/api/ask_ai', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (!resp.ok) throw new Error(`Backend AI error: ${resp.status}`);
        const data = await resp.json();
        const backendText = (data && data.answer) ? String(data.answer) : "";
        const isNegative = /not\s+(?:found|mentioned|covered|available)/i.test(backendText);
        if (backendText && !isNegative) {
          setMessages((prev) => [...prev, { sender: "ai", text: backendText }]);
          return;
        }
      } catch (e) {
        console.warn("[v0] Backend AI unavailable, falling back to Cohere in frontend.", e);
      }

      // Frontend fallback using Cohere directly
      if (!co) {
        setMessages((prev) => [
          ...prev,
          { sender: "ai", text: "AI is offline. Add VITE_COHERE_API_KEY to use AI without backend." },
        ]);
        return;
      }

      const chat = await co.chat({
        model: "command-xlarge-nightly",
        message: input,
        temperature: 0.6,
        maxTokens: 512,
        connectors: [],
      });
      setMessages((prev) => [
        ...prev,
        { sender: "ai", text: chat.text || "No response" },
      ]);
    } catch (error) {
      console.error("[v0] AI Chat error:", error);
      setMessages((prev) => [
        ...prev,
        { sender: "ai", text: "⚠️ Error contacting AI. Make sure the backend server is running." },
      ]);
    }

    setInput("");
    setLoading(false);
  };

  return (
    <div className="ai-chat">
      <h3>AI Assistant</h3>
      <div className="chat-box">
        {messages.map((m, i) => (
          <div
            key={i}
            className={m.sender === "user" ? "message user" : "message ai"}
          >
            {m.text}
          </div>
        ))}
        {loading && <div className="message ai">⏳ Thinking...</div>}
      </div>

      <div className="chat-input">
        <input
          type="text"
          placeholder="Ask a question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
        />
        <button onClick={sendMessage}>➤</button>
      </div>
    </div>
  );
};

export default AiChat;
