"use client"

import type React from "react"
import type { RefObject } from "react"
import { useRef } from "react"

interface Props {
  videoRef: RefObject<HTMLVideoElement>
  onVideoUpload: (file: File) => void
  onTranscribe: () => void
  videoSrc: string | null
  isTranscribing: boolean
}

const VideoPlayer: React.FC<Props> = ({ videoRef, onVideoUpload, onTranscribe, videoSrc, isTranscribing }) => {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && (file.type.startsWith("video/") || file.type.startsWith("audio/"))) {
      onVideoUpload(file)
    } else {
      alert("Please select a valid video or audio file (.mp4, .mp3, .webm, etc.)")
    }
  }

  return (
    <div className="video-section">
      <div className="video-player">
        <video
          ref={videoRef}
          controls
          width="100%"
          src={videoSrc || undefined}
          style={{ display: videoSrc ? "block" : "none" }}
        />
        {!videoSrc && (
          <div className="video-placeholder">
            <p>No video loaded. Click Upload to select a video file.</p>
          </div>
        )}
      </div>
      <div className="video-controls">
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*,audio/*,.mp4,.mp3,.webm,.avi,.mov"
          onChange={handleFileChange}
          style={{ display: "none" }}
        />
        <button className="video-btn secondary" onClick={handleUploadClick}>
          Upload
        </button>
        <button className="video-btn primary" onClick={onTranscribe} disabled={!videoSrc || isTranscribing}>
          {isTranscribing ? "Transcribing..." : "Transcribe"}
        </button>
      </div>
      <div style={{ marginTop: 8 }}>
        <a href="/youtube.html" style={{ textDecoration: "underline", cursor: "pointer" }}>
          do you have youtube url
        </a>
      </div>
    </div>
  )
}

export default VideoPlayer
