import React, { useState, useEffect, useContext } from 'react';
import axios from 'axios';
import { AuthContext } from '../App';
import useScrollAnimations from '../components/ScrollAnimations';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Favorites = () => {
  const [favorites, setFavorites] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { user } = useContext(AuthContext);
  
  useScrollAnimations();

  useEffect(() => {
    fetchFavorites();
  }, []);

  const fetchFavorites = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await axios.get(`${API}/favorites`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      setFavorites(response.data.favorites || []);
    } catch (error) {
      console.error('Error fetching favorites:', error);
      setError('Failed to load favorites');
    } finally {
      setLoading(false);
    }
  };

  const removeFavorite = async (itemId) => {
    try {
      const token = localStorage.getItem('token');
      await axios.delete(`${API}/favorites/${itemId}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      setFavorites(favorites.filter(fav => fav.item_id !== itemId));
    } catch (error) {
      console.error('Error removing favorite:', error);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center py-20">
            <div className="loading-spinner rounded-full h-16 w-16 border-4 border-orange-200 border-t-orange-600 mx-auto mb-6"></div>
            <p className="text-xl text-gray-600">Loading your favorites...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-white py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12 hidden-initially">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            My <span className="text-gradient">Favorites</span>
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Your curated collection of personally selected fashion recommendations
          </p>
        </div>

        {error && (
          <div className="bg-red-50 border-2 border-red-200 rounded-2xl p-8 mb-12 text-center hidden-initially animate-delay-100">
            <div className="text-red-600 text-6xl mb-4">⚠️</div>
            <p className="text-red-800 text-lg mb-4">{error}</p>
            <button
              onClick={fetchFavorites}
              className="bg-red-600 text-white px-6 py-3 rounded-xl hover:bg-red-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        )}

        {favorites.length === 0 ? (
          <div className="text-center py-20 hidden-initially">
            <div className="text-8xl mb-8">💝</div>
            <h3 className="text-3xl font-bold text-gray-900 mb-6">
              No favorites yet
            </h3>
            <p className="text-xl text-gray-600 mb-10 max-w-2xl mx-auto leading-relaxed">
              Start exploring our AI-powered outfit recommendations and save the ones that speak to your style!
            </p>
            <a
              href="/analyze"
              className="btn-primary text-white text-lg inline-block"
            >
              <span className="mr-2">✨</span>
              Discover Perfect Outfits
            </a>
          </div>
        ) : (
          <>
            <div className="text-center mb-8 hidden-initially">
              <p className="text-lg text-gray-600">
                You have <span className="font-bold text-orange-600">{favorites.length}</span> favorite{favorites.length !== 1 ? 's' : ''}
              </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {favorites.map((favorite, index) => (
                <div 
                  key={favorite.id} 
                  className={`outfit-card hidden-initially animate-delay-${(index % 3 + 1) * 100}`}
                >
                  <div className="relative">
                    {/* Product Image */}
                    <div className="image-container h-80">
                      <img
                        src="https://via.placeholder.com/400x500/708090/FFFFFF?text=Fashion+Item"
                        alt={favorite.product_name}
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute inset-0 bg-black bg-opacity-0 hover:bg-opacity-10 transition-all duration-300"></div>
                    </div>
                    
                    {/* Remove Favorite Button */}
                    <button
                      onClick={() => removeFavorite(favorite.item_id)}
                      className="heart-btn absolute top-4 right-4 p-3 rounded-full bg-red-500 text-white hover:bg-red-600 transition-all duration-300 shadow-lg"
                    >
                      <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </button>

                    {/* Favorite Badge */}
                    <div className="absolute top-4 left-4 bg-gradient-to-r from-red-500 to-pink-500 text-white px-3 py-1 rounded-full text-sm font-bold">
                      ❤️ Favorite
                    </div>
                  </div>
                  
                  <div className="p-6">
                    <h3 className="text-xl font-bold text-gray-900 mb-3 line-clamp-2">
                      {favorite.product_name}
                    </h3>
                    
                    <div className="flex items-center justify-between text-sm mb-4">
                      <span className="font-medium text-gray-700">Color:</span>
                      <div className="flex items-center space-x-2">
                        <div 
                          className="color-dot"
                          style={{ backgroundColor: favorite.base_colour.toLowerCase() }}
                        ></div>
                        <span className="text-gray-600">{favorite.base_colour}</span>
                      </div>
                    </div>
                    
                    <div className="text-xs text-gray-500 mb-4">
                      Added on {new Date(favorite.created_at).toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric'
                      })}
                    </div>

                    <button className="w-full bg-gradient-to-r from-gray-800 to-gray-700 text-white py-3 rounded-xl hover:from-gray-700 hover:to-gray-600 transition-all duration-300 transform hover:scale-105 font-medium">
                      View Similar Items
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Additional Actions */}
            <div className="text-center mt-16 hidden-initially animate-delay-500">
              <div className="bg-white rounded-2xl shadow-lg p-8 max-w-2xl mx-auto">
                <h3 className="text-2xl font-bold text-gray-900 mb-4">
                  Want to discover more styles?
                </h3>
                <p className="text-gray-600 mb-6">
                  Explore more personalized recommendations based on your unique style preferences
                </p>
                <a
                  href="/analyze"
                  className="btn-primary text-white text-lg"
                >
                  <span className="mr-2">🎨</span>
                  Get New Recommendations
                </a>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default Favorites;