import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import useScrollAnimations from '../components/ScrollAnimations';

const Home = () => {
  useScrollAnimations();

  const categories = [
    {
      title: 'Elegance',
      image: 'https://images.pexels.com/photos/2876033/pexels-photo-2876033.jpeg',
      description: 'Sophisticated and timeless fashion choices for the modern woman',
      link: '/analyze',
      gradient: 'from-purple-900 to-purple-700'
    },
    {
      title: 'Fashion',
      image: 'https://images.unsplash.com/photo-1483985988355-763728e1935b',
      description: 'Latest trends and contemporary style for every occasion',
      link: '/analyze',
      gradient: 'from-pink-900 to-pink-700'
    },
    {
      title: 'Grace',
      image: 'https://images.pexels.com/photos/1193942/pexels-photo-1193942.jpeg',
      description: 'Effortless beauty and natural charm that speaks volumes',
      link: '/analyze',
      gradient: 'from-indigo-900 to-indigo-700'
    }
  ];

  const features = [
    {
      icon: '📷',
      title: 'Upload Your Photo',
      description: 'Take or upload a clear photo of your face for accurate skin tone analysis',
      delay: 'animate-delay-100'
    },
    {
      icon: '🎨',
      title: 'AI Analysis',
      description: 'Our advanced AI analyzes your skin tone and determines your perfect color palette',
      delay: 'animate-delay-200'
    },
    {
      icon: '💝',
      title: 'Get Recommendations',
      description: 'Receive personalized outfit suggestions and save your favorites',
      delay: 'animate-delay-300'
    }
  ];

  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative h-screen flex items-center justify-center overflow-hidden">
        {/* Background Image */}
        <div 
          className="absolute inset-0 bg-cover bg-center bg-fixed"
          style={{
            backgroundImage: 'url(https://images.unsplash.com/photo-1529139574466-a303027c1d8b)'
          }}
        ></div>
        
        {/* Overlay */}
        <div className="absolute inset-0 hero-overlay"></div>
        
        {/* Content */}
        <div className="relative z-10 text-center text-white px-4 max-w-5xl mx-auto">
          <div className="animate-fade-left">
            <h1 className="text-6xl md:text-8xl font-bold mb-6 leading-tight">
              Discover Your
              <span className="block text-gradient text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 to-orange-400">
                Perfect Style
              </span>
            </h1>
          </div>
          
          <div className="animate-fade-right animate-delay-200">
            <p className="text-xl md:text-2xl mb-12 max-w-3xl mx-auto leading-relaxed opacity-90">
              Get personalized fashion recommendations based on your unique skin tone 
              and style preferences with our AI-powered fashion consultant
            </p>
          </div>
          
          <div className="animate-scale animate-delay-400">
            <Link
              to="/analyze"
              className="btn-primary text-white text-lg inline-block mb-8"
            >
              Start Your Style Journey
            </Link>
          </div>
          
          <div className="animate-fade-left animate-delay-500">
            <div className="flex justify-center items-center space-x-8 text-sm opacity-80">
              <div className="flex items-center">
                <span className="w-2 h-2 bg-yellow-400 rounded-full mr-2"></span>
                AI-Powered Analysis
              </div>
              <div className="flex items-center">
                <span className="w-2 h-2 bg-orange-400 rounded-full mr-2"></span>
                Personalized Results
              </div>
              <div className="flex items-center">
                <span className="w-2 h-2 bg-red-400 rounded-full mr-2"></span>
                Expert Recommendations
              </div>
            </div>
          </div>
        </div>
        
        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 text-white animate-bounce">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
        </div>
      </section>

      {/* Categories Section */}
      <section className="py-24 bg-gradient-to-br from-gray-50 to-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-20 hidden-initially">
            <h2 className="text-5xl font-bold text-gray-900 mb-6">
              Choose Your Style 
              <span className="text-gradient">Category</span>
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
              Explore different fashion aesthetics and discover what truly resonates with your unique personality and lifestyle
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
            {categories.map((category, index) => (
              <div key={index} className={`category-card hidden-initially animate-delay-${(index + 1) * 100}`}>
                <div className="bg-white rounded-3xl shadow-xl overflow-hidden h-full">
                  <div className="relative h-80 overflow-hidden">
                    <img
                      src={category.image}
                      alt={category.title}
                      className="w-full h-full object-cover transition-transform duration-500"
                    />
                    <div className={`absolute inset-0 bg-gradient-to-t ${category.gradient} opacity-60`}></div>
                    <div className="absolute bottom-6 left-6 text-white">
                      <h3 className="text-3xl font-bold mb-2">{category.title}</h3>
                    </div>
                  </div>
                  <div className="p-8">
                    <p className="text-gray-600 mb-6 leading-relaxed">
                      {category.description}
                    </p>
                    <Link
                      to={category.link}
                      className="inline-block bg-gradient-to-r from-gray-800 to-gray-700 text-white px-8 py-3 rounded-full hover:from-gray-700 hover:to-gray-600 transition-all duration-300 transform hover:scale-105 font-medium"
                    >
                      Explore Style →
                    </Link>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 bg-gradient-to-br from-white to-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-20 hidden-initially">
            <h2 className="text-5xl font-bold text-gray-900 mb-6">
              How It <span className="text-gradient">Works</span>
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
              Get personalized fashion recommendations in three simple steps using our advanced AI technology
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
            {features.map((feature, index) => (
              <div key={index} className={`text-center hidden-initially ${feature.delay}`}>
                <div className="bg-gradient-to-br from-orange-100 to-yellow-100 w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-8 text-4xl shadow-lg">
                  {feature.icon}
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4">{feature.title}</h3>
                <p className="text-gray-600 leading-relaxed max-w-sm mx-auto">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900">
        <div className="max-w-4xl mx-auto text-center px-4 sm:px-6 lg:px-8">
          <div className="hidden-initially">
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Ready to Transform Your 
              <span className="text-gradient text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 to-orange-400">
                Wardrobe?
              </span>
            </h2>
            <p className="text-xl text-gray-300 mb-10 leading-relaxed">
              Join thousands of fashion-forward individuals who have discovered their perfect style
            </p>
            <Link
              to="/analyze"
              className="btn-primary text-white text-lg inline-block"
            >
              Get Started Now
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;