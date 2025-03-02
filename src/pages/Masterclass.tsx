import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { BookOpen, Star, Clock, Filter, Search, ArrowRight } from 'lucide-react';

// Mock data for masterclasses
const masterclasses = [
  {
    id: 1,
    title: "The Art of Storytelling in Presentations",
    instructor: "Simon Sinek",
    image: "https://images.unsplash.com/photo-1557804506-669a67965ba0?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80",
    duration: "45 minutes",
    level: "Intermediate",
    category: "Content",
    rating: 4.9,
    description: "Learn how to craft compelling stories that captivate your audience and make your message memorable."
  },
  {
    id: 2,
    title: "Body Language Mastery",
    instructor: "Vinh Giang",
    image: "https://images.unsplash.com/photo-1475721027785-f74eccf877e2?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80",
    duration: "60 minutes",
    level: "All Levels",
    category: "Non-verbal",
    rating: 4.8,
    description: "Discover how to use your body language to enhance your message and connect with your audience."
  },
  {
    id: 3,
    title: "Vocal Power and Presence",
    instructor: "Tony Robbins",
    image: "https://images.unsplash.com/photo-1560439514-4e9645039924?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80",
    duration: "55 minutes",
    level: "Advanced",
    category: "Verbal",
    rating: 4.9,
    description: "Master the techniques to develop a powerful, confident voice that commands attention and respect."
  },
  {
    id: 4,
    title: "Overcoming Presentation Anxiety",
    instructor: "Vinh Giang",
    image: "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80",
    duration: "40 minutes",
    level: "Beginner",
    category: "Mindset",
    rating: 4.7,
    description: "Learn practical techniques to manage nervousness and present with confidence, even under pressure."
  },
  {
    id: 5,
    title: "Creating Impactful Visual Aids",
    instructor: "Simon Sinek",
    image: "https://images.unsplash.com/photo-1581291518633-83b4ebd1d83e?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80",
    duration: "50 minutes",
    level: "Intermediate",
    category: "Content",
    rating: 4.6,
    description: "Design visual aids that enhance your message without overwhelming your audience."
  },
  {
    id: 6,
    title: "The Power of Persuasion",
    instructor: "Tony Robbins",
    image: "https://images.unsplash.com/photo-1551836022-d5d88e9218df?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80",
    duration: "65 minutes",
    level: "Advanced",
    category: "Content",
    rating: 4.9,
    description: "Master the art of persuasive speaking to influence decisions and inspire action."
  }
];

const Masterclass: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [selectedLevel, setSelectedLevel] = useState('All');

  const categories = ['All', 'Content', 'Verbal', 'Non-verbal', 'Mindset'];
  const levels = ['All', 'Beginner', 'Intermediate', 'Advanced'];

  const filteredMasterclasses = masterclasses.filter((masterclass) => {
    const matchesSearch = masterclass.title.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          masterclass.instructor.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'All' || masterclass.category === selectedCategory;
    const matchesLevel = selectedLevel === 'All' || masterclass.level === selectedLevel;
    
    return matchesSearch && matchesCategory && matchesLevel;
  });

  return (
    <div className="bg-gray-50 min-h-screen py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h1 className="text-3xl font-bold text-gray-900">Presentation Masterclasses</h1>
          <p className="mt-4 text-xl text-gray-600 max-w-3xl mx-auto">
            Learn from world-renowned presentation experts and take your skills to the next level.
          </p>
        </div>

        {/* Featured Masterclass */}
        <div className="bg-indigo-700 rounded-xl overflow-hidden shadow-xl mb-12">
          <div className="md:flex">
            <div className="md:w-1/2">
              <img 
                src="https://images.unsplash.com/photo-1475721027785-f74eccf877e2?ixlib=rb-1.2.1&auto=format&fit=crop&w=1050&q=80" 
                alt="Featured Masterclass" 
                className="h-full w-full object-cover"
              />
            </div>
            <div className="md:w-1/2 p-8 md:p-12 text-white">
              <div className="flex items-center mb-4">
                <span className="bg-indigo-500 text-xs font-semibold px-2.5 py-0.5 rounded-full mr-2">FEATURED</span>
                <span className="text-indigo-200 text-sm">By Vinh Giang</span>
              </div>
              <h2 className="text-2xl md:text-3xl font-bold mb-4">Master the Art of Presentation: A Comprehensive Guide</h2>
              <p className="text-indigo-100 mb-6">
                This exclusive masterclass combines the wisdom of three presentation experts to give you a complete toolkit for becoming an exceptional presenter.
              </p>
              <div className="flex items-center mb-6">
                <Clock className="h-5 w-5 mr-2 text-indigo-300" />
                <span className="text-indigo-200 mr-4">90 minutes</span>
                <Star className="h-5 w-5 mr-2 text-yellow-400" />
                <span className="text-indigo-200">4.9 (128 reviews)</span>
              </div>
              <Link
                to="/masterclass/special"
                className="inline-flex items-center justify-center px-5 py-3 border border-transparent text-base font-medium rounded-md text-indigo-700 bg-white hover:bg-indigo-50"
              >
                Watch Now
                <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
            </div>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <div className="md:flex md:items-center md:justify-between">
            <div className="relative flex-1 max-w-lg">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
                placeholder="Search by title or instructor"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            
            <div className="mt-4 md:mt-0 flex flex-wrap items-center gap-4">
              <div className="flex items-center">
                <Filter className="h-5 w-5 text-gray-400 mr-2" />
                <span className="text-sm text-gray-700">Filter by:</span>
              </div>
              
              <select
                className="block w-full md:w-auto pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
              >
                {categories.map((category) => (
                  <option key={category} value={category}>
                    {category}
                  </option>
                ))}
              </select>
              
              <select
                className="block w-full md:w-auto pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
                value={selectedLevel}
                onChange={(e) => setSelectedLevel(e.target.value)}
              >
                {levels.map((level) => (
                  <option key={level} value={level}>
                    {level}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Masterclass Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {filteredMasterclasses.map((masterclass) => (
            <div key={masterclass.id} className="bg-white rounded-lg shadow-md overflow-hidden transition-transform duration-300 hover:shadow-lg hover:-translate-y-1">
              <div className="relative h-48">
                <img
                  src={masterclass.image}
                  alt={masterclass.title}
                  className="w-full h-full object-cover"
                />
                <div className="absolute top-0 right-0 bg-indigo-600 text-white text-xs font-bold px-3 py-1 m-2 rounded-full">
                  {masterclass.category}
                </div>
              </div>
              
              <div className="p-6">
                <div className="flex items-center text-sm text-gray-500 mb-2">
                  <Clock className="h-4 w-4 mr-1" />
                  <span>{masterclass.duration}</span>
                  <span className="mx-2">•</span>
                  <span>{masterclass.level}</span>
                </div>
                
                <h3 className="text-xl font-semibold text-gray-900 mb-2">{masterclass.title}</h3>
                <p className="text-gray-600 mb-4">{masterclass.description}</p>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className="flex items-center">
                      <Star className="h-5 w-5 text-yellow-400" />
                      <span className="ml-1 text-gray-700 font-medium">{masterclass.rating}</span>
                    </div>
                    <span className="mx-2 text-gray-300">|</span>
                    <span className="text-gray-600">By {masterclass.instructor}</span>
                  </div>
                  
                  <Link
                    to={`/masterclass/${masterclass.id}`}
                    className="text-indigo-600 hover:text-indigo-500 font-medium"
                  >
                    View
                  </Link>
                </div>
              </div>
            </div>
          ))}
        </div>

        {filteredMasterclasses.length === 0 && (
          <div className="text-center py-12">
            <BookOpen className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-lg font-medium text-gray-900">No masterclasses found</h3>
            <p className="mt-1 text-gray-500">Try adjusting your search or filter criteria.</p>
          </div>
        )}

        {/* Expert Instructors */}
        <div className="mt-16">
          <h2 className="text-2xl font-bold text-gray-900 mb-8">Our Expert Instructors</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <img
                src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80"
                alt="Simon Sinek"
                className="w-full h-64 object-cover"
              />
              <div className="p-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Simon Sinek</h3>
                <p className="text-gray-600 mb-4">
                  Renowned author and TED speaker known for his expertise in leadership communication and the power of purpose-driven presentations.
                </p>
                <div className="flex items-center justify-between">
                  <span className="text-gray-500">12 Masterclasses</span>
                  <Link
                    to="/masterclass?instructor=Simon%20Sinek"
                    className="text-indigo-600 hover:text-indigo-500 font-medium"
                  >
                    View All
                  </Link>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <img
                src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80"
                alt="Vinh Giang"
                className="w-full h-64 object-cover"
              />
              <div className="p-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Vinh Giang</h3>
                <p className="text-gray-600 mb-4">
                  Magician turned communication expert who teaches the psychology of engagement and non-verbal communication techniques.
                </p>
                <div className="flex items-center justify-between">
                  <span className="text-gray-500">8 Masterclasses</span>
                  <Link
                    to="/masterclass?instructor=Vinh%20Giang"
                    className="text-indigo-600 hover:text-indigo-500 font-medium"
                  >
                    View All
                  </Link>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <img
                src="https://images.unsplash.com/photo-1500648767791-00dcc994a43e?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80"
                alt="Tony Robbins"
                className="w-full h-64 object-cover"
              />
              <div className="p-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Tony Robbins</h3>
                <p className="text-gray-600 mb-4">
                  World-famous motivational speaker and coach who specializes in powerful delivery, audience engagement, and persuasive speaking.
                </p>
                <div className="flex items-center justify-between">
                  <span className="text-gray-500">10 Masterclasses</span>
                  <Link
                    to="/masterclass?instructor=Tony%20Robbins"
                    className="text-indigo-600 hover:text-indigo-500 font-medium"
                  >
                    View All
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* CTA for 1-on-1 Coaching */}
        <div className="mt-16 bg-gradient-to-r from-indigo-600 to-indigo-800 rounded-xl shadow-xl overflow-hidden">
          <div className="md:flex">
            <div className="md:w-2/3 p-8 md:p-12 text-white">
              <h2 className="text-2xl md:text-3xl font-bold mb-4">Want Personalized Coaching?</h2>
              <p className="text-indigo-100 mb-6">
                Schedule a 1-on-1 session with one of our presentation experts for personalized feedback and coaching tailored to your specific needs.
              </p>
              <Link
                to="/schedule"
                className="inline-flex items-center justify-center px-5 py-3 border border-transparent text-base font-medium rounded-md text-indigo-700 bg-white hover:bg-indigo-50"
              >
                Schedule a Session
                <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
            </div>
            <div className="hidden md:block md:w-1/3">
              <img 
                src="https://images.unsplash.com/photo-1522202176988-66273c2fd55f?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80" 
                alt="1-on-1 Coaching" 
                className="h-full w-full object-cover"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Masterclass;