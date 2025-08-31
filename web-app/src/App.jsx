import { useState, useRef, useEffect } from 'react'
import './App.css'

// Use environment variable for API URL
// In production, use relative URLs since frontend and backend are on same domain
// In development, fallback to localhost backend
const API_BASE = import.meta.env.VITE_API_URL || (
  import.meta.env.MODE === 'production' ? '' : 'http://localhost:5001'
)

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [serverConnected, setServerConnected] = useState(false)
  const [checkingServer, setCheckingServer] = useState(true)
  const [error, setError] = useState(null)
  const fileInputRef = useRef(null)

  // Check server status on component mount
  useEffect(() => {
    checkServerStatus()
  }, [])

  const checkServerStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/health`)
      const data = await response.json()
      setServerConnected(data.status === 'healthy' && data.model_loaded)
      setError(null)
    } catch (error) {
      console.error('Server connection failed:', error)
      setServerConnected(false)
      setError('Unable to connect to backend server. Make sure it\'s running on port 5001.')
    } finally {
      setCheckingServer(false)
    }
  }

  const handleImageUpload = (event) => {
    const file = event.target.files[0]
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setSelectedImage(e.target.result)
        setPrediction(null)
        setError(null)
      }
      reader.onerror = () => {
        setError('Failed to read the image file. Please try again.')
      }
      reader.readAsDataURL(file)
    } else if (file && !file.type.startsWith('image/')) {
      setError('Please select a valid image file.')
      // Reset the file input
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const classifyImage = async () => {
    if (!selectedImage) return

    setIsLoading(true)
    setError(null)
    
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: selectedImage
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setPrediction(data)
    } catch (error) {
      console.error('Classification failed:', error)
      setError('Classification failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const resetApp = () => {
    setSelectedImage(null)
    setPrediction(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const getCategoryColor = (category) => {
    const colors = {
      'plastic': '#4a7c95',     // Water light for synthetic materials
      'metal': '#8B7355',       // Earth warm for metals
      'paper': '#A69080',       // Clay soft for paper products
      'shoes': '#8B7355',       // Earth warm for leather/fabric
      'clothes': '#7A8471',     // Sage light for textiles
      'battery': '#A0654B',     // Rust subtle for hazardous
      'biological': '#6B7D5C',  // Leaf fresh for organic
      'cardboard': '#A69080',   // Clay soft for paper-based
      'brown-glass': '#8B7355', // Earth warm
      'green-glass': '#6B7D5C', // Leaf fresh
      'white-glass': '#2c5f7e', // Water mid for clear glass
      'trash': '#8B7355'        // Earth warm
    }
    return colors[category] || '#6B7D5C' // Default to leaf fresh
  }

  const getRecyclingAdvice = (wasteType) => {
    const advice = {
      'battery': 'Take to designated battery recycling centers. Never put in regular trash!',
      'biological': 'Compost in organic waste bin or home composter.',
      'brown-glass': 'Clean and put in brown glass recycling bin.',
      'cardboard': 'Flatten and put in paper recycling bin.',
      'clothes': 'Donate if in good condition, otherwise textile recycling.',
      'green-glass': 'Clean and put in green glass recycling bin.',
      'metal': 'Put in metal recycling bin or scrap metal collection.',
      'paper': 'Clean paper goes in paper recycling bin.',
      'plastic': 'Check recycling number and put in appropriate plastic bin.',
      'shoes': 'Donate if usable, otherwise textile recycling.',
      'trash': 'General waste bin - consider if it can be recycled instead.',
      'white-glass': 'Clean and put in clear glass recycling bin.'
    }
    return advice[wasteType] || 'Please dispose of responsibly.'
  }

  return (
    <div className="App">
      <header className="header">
        <h1>Gather</h1>
        <p>Upload an image to classify waste and get recycling advice</p>
      </header>

      <main className="main">
        {/* Server Status - only show when disconnected */}
        {!serverConnected && !checkingServer && (
          <div className="status disconnected">
            <span>backend disconnected</span>
            <button onClick={checkServerStatus} className="retry-btn">
              retry connection
            </button>
          </div>
        )}

        {error && (
          <div className="error">
            {error}
          </div>
        )}

        {/* Image Upload */}
        <div className="upload-section">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleImageUpload}
            accept="image/*"
            className="file-input"
            id="image-upload"
          />
          <label htmlFor="image-upload" className="upload-btn">
            Choose Image
          </label>
          
          {selectedImage && (
            <button onClick={resetApp} className="reset-btn">
              Clear
            </button>
          )}
        </div>

        {/* Image Preview */}
        {selectedImage && (
          <div className="image-section">
            <img
              src={selectedImage}
              alt="Selected waste item"
              className="preview-image"
            />
            
            <button 
              onClick={classifyImage}
              disabled={isLoading || !serverConnected}
              className="classify-btn"
            >
              {isLoading ? 'Classifying...' : 'Classify Waste'}
            </button>
          </div>
        )}

        {/* Results */}
        {prediction && (
          <div className="results">
            <h2>Classification Results</h2>
            
            <div className="main-result">
              <h3 style={{ color: getCategoryColor(prediction.predicted_class) }}>
                Predicted: {prediction.predicted_class}
              </h3>
              <p className="confidence">
                Confidence: {(prediction.confidence * 100).toFixed(1)}%
              </p>
              
              <div className="advice" style={{ 
                borderLeftColor: getCategoryColor(prediction.predicted_class),
                backgroundColor: `${getCategoryColor(prediction.predicted_class)}20` 
              }}>
                <h4 style={{ color: getCategoryColor(prediction.predicted_class) }}>
                  Recycling Advice:
                </h4>
                <p>{getRecyclingAdvice(prediction.predicted_class)}</p>
              </div>
            </div>

            {prediction.all_predictions && (
              <div className="all-predictions">
                <h4>All Predictions:</h4>
                <div className="predictions-list">
                  {prediction.all_predictions.slice(0, 5).map((pred, index) => (
                    <div key={index} className="prediction-item">
                      <span className="category" style={{ color: getCategoryColor(pred.class) }}>
                        {pred.class}
                      </span>
                      <span className="confidence">
                        {(pred.confidence * 100).toFixed(1)}%
                      </span>
                      <div 
                        className="confidence-bar"
                        style={{ 
                          width: `${pred.confidence * 100}%`,
                          backgroundColor: getCategoryColor(pred.class)
                        }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Building smarter systems for a sustainable future</p>
        <p>© 2024-2025 Gather</p>
      </footer>
    </div>
  )
}

export default App
