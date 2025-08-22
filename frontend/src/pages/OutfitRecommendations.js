import React, { useState, useEffect, useContext } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { AuthContext } from '../App';
import useScrollAnimations from '../components/ScrollAnimations';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const OutfitRecommendations = () => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [gender, setGender] = useState('Men');
  const [favorites, setFavorites] = useState(new Set());
  const location = useLocation();
  const navigate = useNavigate();
  const { user } = useContext(AuthContext);
  
  useScrollAnimations();

  const skinToneData = location.state;

  useEffect(() => {
    if (skinToneData) {
      fetchRecommendations();
    }
    if (user) {
      fetchFavorites();
    }
  }, [skinToneData, gender, user]);

  const fetchRecommendations = async () => {
    setLoading(true);
    setError('');

    try {
      const colors = skinToneData?.recommendedColors?.join(',') || 'Black,White,Blue';
      const response = await axios.get(`${API}/outfit-recommendations`, {
        params: {
          gender: gender,
          recommended_colors: colors,
          limit: 5
        }
      });

      setRecommendations(response.data.recommendations || []);
    } catch (error) {
      console.error('Recommendations error:', error);
      setError('Failed to fetch recommendations. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const fetchFavorites = async () => {
    if (!user) return;

    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(`${API}/favorites`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      const favoriteIds = new Set(response.data.favorites.map(fav => fav.item_id));
      setFavorites(favoriteIds);
    } catch (error) {
      console.error('Error fetching favorites:', error);
    }
  };

  const toggleFavorite = async (item) => {
    if (!user) {
      navigate('/login');
      return;
    }

    const token = localStorage.getItem('token');
    const isCurrentlyFavorite = favorites.has(item.item_id);

    try {
      if (isCurrentlyFavorite) {
        await axios.delete(`${API}/favorites/${item.item_id}`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        setFavorites(prev => {
          const newSet = new Set(prev);
          newSet.delete(item.item_id);
          return newSet;
        });
      } else {
        await axios.post(`${API}/favorites`, {
          item_id: item.item_id,
          product_name: item.product_name,
          base_colour: item.base_colour
        }, {
          headers: { Authorization: `Bearer ${token}` }
        });
        setFavorites(prev => new Set([...prev, item.item_id]));
      }
    } catch (error) {
      console.error('Error toggling favorite:', error);
    }
  };

  const handleAnalyzeNew = () => {
    navigate('/analyze');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-12 hidden-initially">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            Your Perfect <span className="text-gradient">Outfit</span> Recommendations
          </h1>
          {skinToneData && (
            <div className="mb-8">
              <p className="text-xl text-gray-600 mb-4">
                Curated specifically for your unique skin tone
              </p>
              <div className="flex items-center justify-center space-x-6 bg-white rounded-2xl p-6 shadow-lg max-w-2xl mx-auto">
                <div className="flex items-center space-x-3">
                  <span className="text-sm font-medium text-gray-700">Your Skin Tone:</span>
                  <div
                    className="w-8 h-8 rounded-full border-2 border-gray-300 shadow-md"
                    style={{ backgroundColor: skinToneData.skinTone }}
                  ></div>
                </div>
                <div className="flex flex-wrap gap-2">
                  {skinToneData.recommendedColors?.slice(0, 4).map((color, index) => (
                    <span
                      key={index}
                      className="bg-gradient-to-r from-orange-100 to-yellow-100 text-orange-800 px-3 py-1 rounded-full text-sm font-medium"
                    >
                      {color}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Gender Selection */}
        <div className="bg-white rounded-2xl shadow-lg p-8 mb-12 hidden-initially animate-delay-100">
          <h3 className="text-2xl font-bold text-gray-900 mb-6 text-center">Select Your Style Preference</h3>
          <div className="flex justify-center space-x-6">
            {['Men', 'Women'].map((g) => (
              <button
                key={g}
                onClick={() => setGender(g)}
                className={`px-8 py-4 rounded-2xl font-medium transition-all duration-300 transform hover:scale-105 ${
                  gender === g
                    ? 'bg-gradient-to-r from-gray-800 to-gray-700 text-white shadow-lg'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {g}'s Fashion
              </button>
            ))}
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="text-center py-20">
            <div className="loading-spinner rounded-full h-16 w-16 border-4 border-orange-200 border-t-orange-600 mx-auto mb-6"></div>
            <p className="text-xl text-gray-600">Curating perfect outfits for you...</p>
            <div className="max-w-xs mx-auto mt-4">
              <div className="progress-bar w-full"></div>
            </div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-50 border-2 border-red-200 rounded-2xl p-8 mb-12 text-center">
            <div className="text-red-600 text-6xl mb-4">⚠️</div>
            <p className="text-red-800 text-lg mb-4">{error}</p>
            <button
              onClick={fetchRecommendations}
              className="bg-red-600 text-white px-6 py-3 rounded-xl hover:bg-red-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        )}

        {/* Recommendations Grid */}
        {!loading && recommendations.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
            {recommendations.map((item, index) => (
              <div 
                key={item.item_id} 
                className={`outfit-card hidden-initially animate-delay-${(index + 1) * 100}`}
              >
                <div className="relative">
                  {/* Product Image */}
                  <div className="image-container h-80">
                    <img
                      src={item.image_url}
                      alt={item.product_name}
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute inset-0 bg-black bg-opacity-0 hover:bg-opacity-10 transition-all duration-300"></div>
                  </div>
                  
                  {/* Favorite Button */}
                  <button
                    onClick={() => toggleFavorite(item)}
                    className={`heart-btn absolute top-4 right-4 p-3 rounded-full transition-all duration-300 ${
                      favorites.has(item.item_id)
                        ? 'bg-red-500 text-white favorited'
                        : 'bg-white bg-opacity-90 text-gray-400 hover:text-red-500 hover:bg-opacity-100'
                    } shadow-lg`}
                  >
                    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clipRule="evenodd" />
                    </svg>
                  </button>

                  {/* Price Tag */}
                  <div className="absolute top-4 left-4 bg-gradient-to-r from-orange-500 to-red-500 text-white px-3 py-1 rounded-full text-sm font-bold">
                    {item.usage}
                  </div>
                </div>
                
                <div className="p-6">
                  <h3 className="text-xl font-bold text-gray-900 mb-3 line-clamp-2">
                    {item.product_name}
                  </h3>
                  
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium text-gray-700">Category:</span>
                      <span className="text-gray-600">{item.category} - {item.sub_category}</span>
                    </div>
                    
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium text-gray-700">Type:</span>
                      <span className="text-gray-600">{item.article_type}</span>
                    </div>
                    
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium text-gray-700">Color:</span>
                      <div className="flex items-center space-x-2">
                        <div 
                          className="color-dot"
                          style={{ backgroundColor: item.base_colour.toLowerCase() }}
                        ></div>
                        <span className="text-gray-600">{item.base_colour}</span>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium text-gray-700">Season:</span>
                      <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs">
                        {item.season}
                      </span>
                    </div>
                  </div>

                  <button className="w-full mt-6 bg-gradient-to-r from-gray-800 to-gray-700 text-white py-3 rounded-xl hover:from-gray-700 hover:to-gray-600 transition-all duration-300 transform hover:scale-105 font-medium">
                    View Details
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* No Recommendations */}
        {!loading && recommendations.length === 0 && (
          <div className="text-center py-20">
            <div className="text-8xl mb-6">🛍️</div>
            <h3 className="text-3xl font-bold text-gray-900 mb-6">
              No recommendations found
            </h3>
            <p className="text-xl text-gray-600 mb-10 max-w-2xl mx-auto">
              We couldn't find any outfits matching your criteria. Try analyzing your skin tone first or adjust your preferences.
            </p>
            <button
              onClick={handleAnalyzeNew}
              className="btn-primary text-white text-lg"
            >
              Analyze Skin Tone
            </button>
          </div>
        )}

        {/* Action Buttons */}
        {recommendations.length > 0 && (
          <div className="text-center hidden-initially animate-delay-500">
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h3 className="text-2xl font-bold text-gray-900 mb-4">
                Love these recommendations?
              </h3>
              <p className="text-gray-600 mb-6">
                Get more personalized suggestions or try a different analysis
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <button
                  onClick={handleAnalyzeNew}
                  className="bg-gradient-to-r from-gray-800 to-gray-700 text-white px-8 py-3 rounded-xl hover:from-gray-700 hover:to-gray-600 transition-all duration-300 font-medium"
                >
                  Analyze Another Photo
                </button>
                {user && (
                  <button
                    onClick={() => navigate('/favorites')}
                    className="bg-gradient-to-r from-red-500 to-pink-500 text-white px-8 py-3 rounded-xl hover:from-red-600 hover:to-pink-600 transition-all duration-300 font-medium"
                  >
                    View My Favorites
                  </button>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default OutfitRecommendations;