"use client"

import type React from "react"

interface Props {
  query: string
  setQuery: (q: string) => void
  onSearch: () => void
  disabled: boolean
}

const SearchBar: React.FC<Props> = ({ query, setQuery, onSearch, disabled }) => {
  return (
    <div className="search-bar">
      <input
        type="text"
        placeholder="Search for topics, concepts, or questions in your video..."
        value={query}
        onChange={(e) => {
          console.log("[v0] SearchBar input changed:", e.target.value)
          setQuery(e.target.value)
        }}
        disabled={disabled}
      />
      <button onClick={onSearch} disabled={disabled}>
        Search
      </button>
    </div>
  )
}

export default SearchBar
