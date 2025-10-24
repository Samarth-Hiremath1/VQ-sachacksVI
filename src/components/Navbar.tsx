import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Mic, BarChart2, BookOpen, Calendar, Home } from 'lucide-react';

const Navbar: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  const toggleMenu = () => {
    setIsOpen(!isOpen);
  };

  const navLinks = [
    { name: 'Home', path: '/', icon: <Home className="w-5 h-5 mr-2" /> },
    { name: 'Record', path: '/record', icon: <Mic className="w-5 h-5 mr-2" /> },
    { name: 'Dashboard', path: '/dashboard', icon: <BarChart2 className="w-5 h-5 mr-2" /> },
    { name: 'Masterclass', path: '/masterclass', icon: <BookOpen className="w-5 h-5 mr-2" /> },
    { name: 'Schedule', path: '/schedule', icon: <Calendar className="w-5 h-5 mr-2" /> },
  ];

  return (
    <nav className="bg-indigo-700 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center">
              <Mic className="h-8 w-8 mr-2" />
              <span className="font-bold text-xl">VQ</span>
            </Link>
          </div>
          
          {/* Desktop menu */}
          <div className="hidden md:flex items-center space-x-4">
            {navLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                className={`flex items-center px-3 py-2 rounded-md text-sm font-medium ${
                  location.pathname === link.path
                    ? 'bg-indigo-800 text-white'
                    : 'text-indigo-100 hover:bg-indigo-600'
                }`}
              >
                {link.icon}
                {link.name}
              </Link>
            ))}
          </div>
          
          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={toggleMenu}
              className="inline-flex items-center justify-center p-2 rounded-md text-indigo-100 hover:text-white hover:bg-indigo-600 focus:outline-none"
            >
              {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
        </div>
      </div>
      
      {/* Mobile menu */}
      {isOpen && (
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            {navLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                className={`flex items-center px-3 py-2 rounded-md text-sm font-medium ${
                  location.pathname === link.path
                    ? 'bg-indigo-800 text-white'
                    : 'text-indigo-100 hover:bg-indigo-600'
                }`}
                onClick={() => setIsOpen(false)}
              >
                {link.icon}
                {link.name}
              </Link>
            ))}
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;