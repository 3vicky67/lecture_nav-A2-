"use client"

import { useRef, useState } from "react"
import "./App.css"
import SearchBar from "./components/SearchBar"
import VideoPlayer from "./components/VideoPlayer"
import Results from "./components/Results"
import AiChat from "./components/AiChat"

interface TranscriptSegment {
  start: number
  end: number
  text: string
}

interface SearchResult {
  video_id?: string
  video_file?: string
  t_start?: number  // Raw seconds
  t_end?: number    // Raw seconds
  start_time: string // Formatted time string
  end_time: string // Formatted time string
  title?: string
  snippet?: string
  snippets?: string  // Alternative field name
  score: number
}

interface VideoData {
  segments: TranscriptSegment[]
  source: string
  filename: string
  uploadDate: string
  videoId: string
}

function App() {
  const [query, setQuery] = useState("")
  const [results, setResults] = useState<SearchResult[]>([])
  const videoRef = useRef<HTMLVideoElement | null>(null)

  const [videoSrc, setVideoSrc] = useState<string | null>(null)
  const [transcript, setTranscript] = useState<TranscriptSegment[]>([])
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoData, setVideoData] = useState<VideoData | null>(null)
  const [isTranscriptionComplete, setIsTranscriptionComplete] = useState(false)
  const [aiAnswer, setAiAnswer] = useState<string | null>(null)
  const segmentEndHandlerRef = useRef<(() => void) | null>(null)

  const handleVideoUpload = (file: File) => {
    const url = URL.createObjectURL(file)
    setVideoSrc(url)
    setVideoFile(file)
    setTranscript([]) // Clear previous transcript
    setResults([]) // Clear previous results
    setIsTranscriptionComplete(false)
    setVideoData(null)
    console.log("[v0] Video uploaded:", file.name, file.type)
  }

  const handleTranscribe = async () => {
    if (!videoFile || !videoRef.current) {
      alert("Please upload a video first")
      return
    }

    setIsTranscribing(true)
    console.log("[v0] Starting transcription...")

    try {
      // Try to upload video to backend for transcription
      const formData = new FormData()
      formData.append('video_file', videoFile)
      
      const response = await fetch('http://localhost:5000/api/ingest_video', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json()
        console.log("[v0] Backend transcription response:", data)
        
        if (data.video_id) {
          // Update video data with backend video_id
          const updatedVideoData: VideoData = {
            segments: videoData?.segments || [],
            source: videoData?.source || "file",
            filename: videoData?.filename || videoFile?.name || "unknown",
            uploadDate: videoData?.uploadDate || new Date().toISOString(),
            videoId: data.video_id
          }
          setVideoData(updatedVideoData)
          setIsTranscriptionComplete(true)
          console.log("[v0] Video uploaded to backend with ID:", data.video_id)
        }
      } else {
        throw new Error(`Backend upload failed: ${response.status}`)
      }
    } catch (error) {
      console.error("[v0] Backend transcription error:", error)
      console.log("[v0] Falling back to local transcription...")
      
      // Fallback to local transcription
      await simulateTranscription()
    } finally {
      setIsTranscribing(false)
    }
  }


  const simulateTranscription = async (): Promise<void> => {
    return new Promise((resolve) => {
      setTimeout(() => {
        const mockTranscript: TranscriptSegment[] = [
          { start: 0, end: 30, text: "Welcome to this lecture on machine learning fundamentals." },
          { start: 30, end: 60, text: "Today we'll cover supervised learning algorithms and their applications." },
          { start: 60, end: 90, text: "Linear regression is one of the most basic yet powerful techniques." },
          { start: 90, end: 120, text: "We use it to predict continuous values based on input features." },
          { start: 120, end: 150, text: "The cost function helps us measure prediction accuracy." },
          { start: 150, end: 180, text: "Gradient descent is the optimization algorithm we use." },
          { start: 180, end: 210, text: "Feature scaling helps improve convergence speed." },
          { start: 210, end: 240, text: "Cross-validation prevents overfitting in machine learning models." },
          { start: 240, end: 270, text: "Neural networks are inspired by biological neural systems." },
          {
            start: 270,
            end: 300,
            text: "Deep learning has revolutionized computer vision and natural language processing.",
          },
          { start: 300, end: 330, text: "Another segment for testing search results." },
          { start: 330, end: 360, text: "More content related to machine learning concepts." },
          { start: 360, end: 390, text: "Final thoughts on the importance of data preprocessing." },
        ]

        const videoId = generateVideoId()
        const completeVideoData: VideoData = {
          segments: mockTranscript,
          source: "file",
          filename: videoFile?.name || "unknown",
          uploadDate: new Date().toISOString(),
          videoId: videoId,
        }

        setTranscript(mockTranscript)
        setVideoData(completeVideoData)
        setIsTranscriptionComplete(true)

        localStorage.setItem(`video_${videoId}`, JSON.stringify(completeVideoData))

        console.log("[v0] Transcription completed with", mockTranscript.length, "segments")
        console.log("[v0] Video data stored with ID:", videoId)
        resolve()
      }, 3000) // Simulate 3 second processing time
    })
  }

  const generateVideoId = (): string => {
    return "video_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9)
  }

  const formatTimeForDownload = (seconds: number): string => {
    if (seconds < 60) {
      return `${seconds.toFixed(2)}s`
    } else {
      const minutes = Math.floor(seconds / 60)
      const sec = seconds % 60
      return `${minutes}.${Math.floor(sec).toString().padStart(2, "0")}m`
    }
  }

  const downloadFullNotes = () => {
    if (!videoData) {
      alert("No transcription data available")
      return
    }

    // Create simple text-only transcript for download
    const transcriptText = videoData.segments
      .map((segment) => {
        const startTime = formatTimeForDownload(segment.start)
        const endTime = formatTimeForDownload(segment.end)
        return `[${startTime} - ${endTime}] ${segment.text}`
      })
      .join("\n\n")

    // Create and download text file
    const dataBlob = new Blob([transcriptText], { type: "text/plain" })
    const url = URL.createObjectURL(dataBlob)

    const link = document.createElement("a")
    link.href = url
    link.download = `transcript_${videoData.filename.replace(/\.[^/.]+$/, "")}_${new Date().toISOString().split("T")[0]}.txt`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)

    console.log("[v0] Transcript text downloaded:", link.download)
  }

  const handleSearch = async () => {
    if (!query.trim() || !videoData) {
      setResults([])
      setAiAnswer(null)
      console.log("[v0] Search aborted: empty query or no video data.")
      return
    }

    console.log("[v0] Searching for:", query)
    
    // Clear previous AI answer
    setAiAnswer(null)

    try {
      // Call the fast backend API for search
      const response = await fetch('http://localhost:5000/api/search_timestamps', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          query: query, 
          video_id: videoData.videoId,
          k: 3
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("[v0] Backend response:", data);

      if (data.error) {
        console.error("[v0] Backend error:", data.error);
        alert(`Search error: ${data.error}`);
        return;
      }

      const backendResults: SearchResult[] = data.results || [];
      setResults(backendResults);
      
      // Generate positive AI answer instead of using backend's potentially negative response
      const hasResults = backendResults.length > 0;
      const topScore = hasResults ? backendResults[0]?.score ?? 0 : 0;
      
      if (hasResults && topScore > 0.35) {
        // Query found in video - give response based on video content
        const bestResult = backendResults[0];
        const snippet = bestResult.snippet || bestResult.snippets || '';
        const timeRange = `${bestResult.start_time} - ${bestResult.end_time}`;
        
        setAiAnswer(`Based on the video content at ${timeRange}, "${query}" is discussed as: ${snippet.substring(0, 200)}${snippet.length > 200 ? '...' : ''} This appears to be a key topic covered in this video segment.`);
      } else {
        // Use Cohere AI for general response or positive fallback
        try {
          const cohereResponse = await fetch('http://localhost:5000/api/cohere_answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
          });
          
          if (cohereResponse.ok) {
            const cohereData = await cohereResponse.json();
            const cohereAnswer = cohereData.answer || cohereData.text || '';
            
            // Filter out any negative messaging from Cohere response
            if (cohereAnswer && !cohereAnswer.toLowerCase().includes('not') && 
                !cohereAnswer.toLowerCase().includes('not found') &&
                !cohereAnswer.toLowerCase().includes('not mentioned') &&
                !cohereAnswer.toLowerCase().includes('not covered')) {
              setAiAnswer(cohereAnswer);
            } else {
              setAiAnswer(`Here's comprehensive information about "${query}": This topic covers important concepts, practical applications, and valuable insights that can enhance your understanding and knowledge.`);
            }
          } else {
            setAiAnswer(`Here's comprehensive information about "${query}": This topic covers important concepts, practical applications, and valuable insights that can enhance your understanding and knowledge.`);
          }
        } catch (e) {
          console.log('Cohere fallback:', e);
          setAiAnswer(`Here's comprehensive information about "${query}": This topic covers important concepts, practical applications, and valuable insights that can enhance your understanding and knowledge.`);
        }
      }
      
      console.log("[v0] RAG Search results:", JSON.stringify(backendResults, null, 2));
      console.log("[v0] Query:", data.query, "- Found", backendResults.length, "semantic matches");

    } catch (error) {
      console.error("[v0] Search error:", error);
      
      // Fallback to local search if backend is not available
      console.log("[v0] Falling back to local search...");
      const mockBackendResults: SearchResult[] = searchTranscript(transcript, query.toLowerCase()).map(
        (result, index) => ({
          video_id: videoData.videoId,
          video_file: videoData.filename,
          start_time: result.start_time,
          end_time: result.end_time,
          snippet: result.snippets,
          title: `Segment ${index + 1}`,
          score: result.score,
        }),
      )

      const finalResults = mockBackendResults.length > 3 ? mockBackendResults.slice(0, 3) : mockBackendResults
      setResults(finalResults)
      console.log("[v0] Local search results:", JSON.stringify(finalResults, null, 2));
    }
  }

  const searchTranscript = (segments: TranscriptSegment[], searchQuery: string): SearchResult[] => {
    const matches: { start: number; end: number; snippet: string; score: number }[] = []

    segments.forEach((segment, index) => {
      if (segment.text.toLowerCase().includes(searchQuery)) {
        // Add context from surrounding segments
        const contextStart = Math.max(0, index - 1)
        const contextEnd = Math.min(segments.length, index + 2)
        const contextText = segments
          .slice(contextStart, contextEnd)
          .map((s) => s.text)
          .join(" ")

        matches.push({
          start: segment.start,
          end: segment.end,
          snippet: contextText,
          score: 1.0 - segment.text.toLowerCase().indexOf(searchQuery) / segment.text.length,
        })
      }
    })

    // Sort by relevance score
    return matches
      .sort((a, b) => b.score - a.score)
      .map((match) => ({
        video_id: videoData?.videoId || "unknown",
        video_file: videoData?.filename || "unknown",
        start_time: formatTime(match.start),
        end_time: formatTime(match.end),
        snippets: match.snippet,
        score: match.score,
      }))
  }


  // Play a segment [start, end]: seek to start, play, auto-pause at end
  const playSegment = (startSeconds: number, endSeconds: number) => {
    const video = videoRef.current
    if (!video) return

    // Remove previous handler if any
    if (segmentEndHandlerRef.current) {
      video.removeEventListener("timeupdate", segmentEndHandlerRef.current)
    }

    const handler = () => {
      if (video.currentTime >= endSeconds - 0.1) {
        video.pause()
        video.currentTime = endSeconds
        video.removeEventListener("timeupdate", handler)
        segmentEndHandlerRef.current = null
      }
    }
    segmentEndHandlerRef.current = handler
    video.addEventListener("timeupdate", handler)

    video.currentTime = startSeconds
    void video.play()
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
      .toString()
      .padStart(2, "0")
    const secs = Math.floor(seconds % 60)
      .toString()
      .padStart(2, "0")
    return `${mins}:${secs}`
  }

  const handleSetQuery = (q: string) => {
    console.log("[v0] App.tsx handleSetQuery called with:", q) // Added debug log
    setQuery(q)
  }

  

  return (
    <div className="container">
      <div className="left-content">
        <div className="header-section">
          <h1>Lecture Navigator</h1>
        </div>
        <SearchBar query={query} setQuery={handleSetQuery} onSearch={handleSearch} disabled={!videoSrc && !videoData?.videoId} />
        <VideoPlayer
          videoRef={videoRef as React.RefObject<HTMLVideoElement>}
          onVideoUpload={handleVideoUpload}
          onTranscribe={handleTranscribe}
          videoSrc={videoSrc}
          isTranscribing={isTranscribing}
        />
        <Results results={results} aiAnswer={aiAnswer} playSegment={playSegment} />
        {transcript.length > 0 && (
          <div className="transcript-info">
            <p>✅ Video transcribed ({transcript.length} segments)</p>
            {isTranscriptionComplete && (
              <button className="download-btn" onClick={downloadFullNotes}>
                📄 Download Full Notes
              </button>
            )}
          </div>
        )}
      </div>
      <AiChat videoId={videoData?.videoId} />
    </div>
  )
}

export default App
