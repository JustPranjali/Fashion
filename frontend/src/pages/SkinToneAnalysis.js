import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import useScrollAnimations from '../components/ScrollAnimations';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const SkinToneAnalysis = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();
  
  useScrollAnimations();

  const handleFileSelect = (file) => {
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please select a valid image file (JPG, PNG, or GIF)');
        return;
      }
      
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        setError('File size must be less than 10MB');
        return;
      }
      
      setSelectedFile(file);
      setError('');
      
      // Create preview URL
      const reader = new FileReader();
      reader.onload = (e) => setPreviewUrl(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    handleFileSelect(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setAnalyzing(true);
    setError('');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const token = localStorage.getItem('token');
      const headers = token ? { Authorization: `Bearer ${token}` } : {};
      
      const response = await axios.post(`${API}/analyze-skin-tone`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          ...headers
        }
      });

      setResult(response.data);
    } catch (error) {
      console.error('Analysis error:', error);
      setError(error.response?.data?.detail || 'Analysis failed. Please try again with a clearer photo.');
    } finally {
      setLoading(false);
      setAnalyzing(false);
    }
  };

  const handleGetRecommendations = () => {
    if (result) {
      navigate('/recommendations', {
        state: {
          skinTone: result.detected_skin_tone,
          recommendedColors: result.recommended_colors
        }
      });
    }
  };

  const resetAnalysis = () => {
    setResult(null);
    setPreviewUrl('');
    setSelectedFile(null);
    setError('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white py-12">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12 hidden-initially">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            AI Skin Tone <span className="text-gradient">Analysis</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Upload your photo to discover your unique skin tone and unlock personalized color recommendations 
            that will enhance your natural beauty
          </p>
        </div>

        <div className="bg-white rounded-3xl shadow-xl p-8 lg:p-12">
          {!result ? (
            <div className="space-y-10">
              {/* File Upload Section */}
              <div className="text-center hidden-initially animate-delay-100">
                <div
                  className={`upload-area ${dragOver ? 'drag-over' : ''} p-12 cursor-pointer`}
                  onClick={() => fileInputRef.current?.click()}
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                >
                  {previewUrl ? (
                    <div className="space-y-6">
                      <div className="relative inline-block">
                        <img
                          src={previewUrl}
                          alt="Preview"
                          className="mx-auto h-80 w-80 object-cover rounded-2xl shadow-lg"
                        />
                        <div className="absolute inset-0 rounded-2xl bg-black bg-opacity-0 hover:bg-opacity-10 transition-all duration-300"></div>
                      </div>
                      <p className="text-gray-600 font-medium">Click to change image</p>
                    </div>
                  ) : (
                    <div className="space-y-6">
                      <div className="text-6xl mb-4">📸</div>
                      <div>
                        <p className="text-2xl font-bold text-gray-900 mb-2">
                          Upload Your Photo
                        </p>
                        <p className="text-gray-600">
                          Drag & drop your image here, or click to browse
                        </p>
                        <p className="text-sm text-gray-500 mt-2">
                          Supports PNG, JPG, GIF up to 10MB
                        </p>
                      </div>
                    </div>
                  )}
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </div>

              {/* Instructions */}
              <div className="bg-gradient-to-r from-orange-50 to-yellow-50 rounded-2xl p-8 hidden-initially animate-delay-200">
                <h3 className="text-2xl font-bold text-orange-900 mb-6 text-center">
                  📋 For Best Results
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="flex items-start space-x-3">
                      <span className="text-orange-500 text-xl">💡</span>
                      <span className="text-orange-800">Use natural lighting or bright indoor light</span>
                    </div>
                    <div className="flex items-start space-x-3">
                      <span className="text-orange-500 text-xl">👤</span>
                      <span className="text-orange-800">Face the camera directly with a clear view</span>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div className="flex items-start space-x-3">
                      <span className="text-orange-500 text-xl">🧼</span>
                      <span className="text-orange-800">Remove or minimize makeup if possible</span>
                    </div>
                    <div className="flex items-start space-x-3">
                      <span className="text-orange-500 text-xl">🎯</span>
                      <span className="text-orange-800">Ensure good image quality and focus</span>
                    </div>
                  </div>
                </div>
              </div>

              {error && (
                <div className="bg-red-50 border-2 border-red-200 rounded-2xl p-6 hidden-initially animate-delay-300">
                  <div className="flex items-center space-x-3">
                    <span className="text-red-500 text-2xl">⚠️</span>
                    <p className="text-red-800 font-medium">{error}</p>
                  </div>
                </div>
              )}

              {/* Analyze Button */}
              <div className="text-center hidden-initially animate-delay-400">
                <button
                  onClick={handleAnalyze}
                  disabled={loading || !selectedFile}
                  className="btn-primary text-white text-lg disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {analyzing ? (
                    <div className="flex items-center justify-center">
                      <div className="loading-spinner w-6 h-6 border-2 border-white border-t-transparent rounded-full mr-3"></div>
                      Analyzing Your Skin Tone...
                    </div>
                  ) : (
                    <>
                      <span className="mr-2">🎨</span>
                      Analyze My Skin Tone
                    </>
                  )}
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-10">
              {/* Results Section */}
              <div className="text-center hidden-initially">
                <h2 className="text-4xl font-bold text-gray-900 mb-8">
                  ✨ Analysis Complete!
                </h2>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
                  {/* Image */}
                  <div className="hidden-initially animate-delay-100">
                    <div className="relative inline-block">
                      <img
                        src={previewUrl}
                        alt="Analyzed"
                        className="w-full max-w-md h-96 object-cover rounded-2xl shadow-lg mx-auto"
                      />
                      <div className="absolute -top-3 -right-3 bg-green-500 text-white p-3 rounded-full shadow-lg">
                        <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                      </div>
                    </div>
                  </div>
                  
                  {/* Results */}
                  <div className="space-y-8 hidden-initially animate-delay-200">
                    <div className="bg-gradient-to-br from-gray-50 to-white p-8 rounded-2xl shadow-lg">
                      <h3 className="text-2xl font-bold text-gray-900 mb-4 text-center">
                        Your Skin Tone
                      </h3>
                      <div className="flex items-center justify-center space-x-4 mb-6">
                        <div
                          className="w-16 h-16 rounded-full border-4 border-gray-300 shadow-lg"
                          style={{ backgroundColor: result.detected_skin_tone }}
                        ></div>
                        <div>
                          <span className="text-xl font-mono font-bold text-gray-800">
                            {result.detected_skin_tone}
                          </span>
                          <p className="text-gray-600">Detected Color</p>
                        </div>
                      </div>
                    </div>
                    
                    <div className="bg-gradient-to-br from-orange-50 to-yellow-50 p-8 rounded-2xl">
                      <h3 className="text-2xl font-bold text-orange-900 mb-6 text-center">
                        Perfect Colors for You
                      </h3>
                      <div className="grid grid-cols-2 gap-3">
                        {result.recommended_colors.map((color, index) => (
                          <span
                            key={index}
                            className="bg-white text-orange-800 px-4 py-3 rounded-xl text-center font-medium shadow-md hover:shadow-lg transition-shadow"
                          >
                            {color}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="text-center space-y-6 hidden-initially animate-delay-300">
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <button
                    onClick={handleGetRecommendations}
                    className="btn-primary text-white text-lg"
                  >
                    <span className="mr-2">👗</span>
                    Get My Outfit Recommendations
                  </button>
                  <button
                    onClick={resetAnalysis}
                    className="bg-gray-200 text-gray-700 px-8 py-4 rounded-full font-medium hover:bg-gray-300 transition-colors"
                  >
                    <span className="mr-2">🔄</span>
                    Analyze Another Photo
                  </button>
                </div>
                
                <p className="text-gray-600 max-w-2xl mx-auto">
                  Ready to discover outfits that complement your unique skin tone? 
                  Our AI will curate personalized fashion recommendations just for you!
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SkinToneAnalysis;