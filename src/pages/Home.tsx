import React from 'react';
import { Link } from 'react-router-dom';
import { Mic, BarChart2, BookOpen, Calendar, ArrowRight, CheckCircle } from 'lucide-react';

const Home: React.FC = () => {
  return (
    <div className="bg-white">
      {/* Hero Section */}
      <section className="relative bg-indigo-700 text-white">
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-900 to-indigo-700 opacity-90"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24 md:py-32">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
            <div>
              <h1 className="text-4xl md:text-5xl font-bold leading-tight">
                Master the Art of Presentation with AI-Powered Feedback
              </h1>
              <p className="mt-6 text-xl text-indigo-100">
                Record your presentations, receive instant analysis, and improve with personalized coaching from industry experts.
              </p>
              <div className="mt-8 flex flex-col sm:flex-row gap-4">
                <Link
                  to="/record"
                  className="inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-indigo-700 bg-white hover:bg-indigo-50"
                >
                  Start Recording
                  <Mic className="ml-2 h-5 w-5" />
                </Link>
                <Link
                  to="/masterclass"
                  className="inline-flex items-center justify-center px-6 py-3 border border-white text-base font-medium rounded-md text-white hover:bg-indigo-600"
                >
                  Explore Masterclasses
                  <BookOpen className="ml-2 h-5 w-5" />
                </Link>
              </div>
            </div>
            <div className="hidden md:block">
              <img
                src="https://images.unsplash.com/photo-1557804506-669a67965ba0?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80"
                alt="Person giving presentation"
                className="rounded-lg shadow-xl"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 md:py-24 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-bold text-gray-900">How It Works</h2>
            <p className="mt-4 text-xl text-gray-600 max-w-3xl mx-auto">
              Our platform combines cutting-edge AI analysis with expert coaching to help you become a confident presenter.
            </p>
          </div>

          <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-white p-8 rounded-lg shadow-md">
              <div className="bg-indigo-100 p-3 rounded-full w-14 h-14 flex items-center justify-center">
                <Mic className="h-8 w-8 text-indigo-600" />
              </div>
              <h3 className="mt-6 text-xl font-semibold text-gray-900">Record Your Presentation</h3>
              <p className="mt-4 text-gray-600">
                Use our platform to record yourself presenting. Our system captures both video and audio for comprehensive analysis.
              </p>
              <Link to="/record" className="mt-6 inline-flex items-center text-indigo-600 hover:text-indigo-500">
                Start Recording <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </div>

            <div className="bg-white p-8 rounded-lg shadow-md">
              <div className="bg-indigo-100 p-3 rounded-full w-14 h-14 flex items-center justify-center">
                <BarChart2 className="h-8 w-8 text-indigo-600" />
              </div>
              <h3 className="mt-6 text-xl font-semibold text-gray-900">Get AI Analysis</h3>
              <p className="mt-4 text-gray-600">
                Receive detailed feedback on your speaking pace, clarity, body language, filler words, and overall presentation structure.
              </p>
              <Link to="/dashboard" className="mt-6 inline-flex items-center text-indigo-600 hover:text-indigo-500">
                View Dashboard <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </div>

            <div className="bg-white p-8 rounded-lg shadow-md">
              <div className="bg-indigo-100 p-3 rounded-full w-14 h-14 flex items-center justify-center">
                <Calendar className="h-8 w-8 text-indigo-600" />
              </div>
              <h3 className="mt-6 text-xl font-semibold text-gray-900">Schedule Expert Coaching</h3>
              <p className="mt-4 text-gray-600">
                Book 1-on-1 sessions with presentation experts like Vinh Giang, Simon Sinek, and Tony Robbins for personalized guidance.
              </p>
              <Link to="/schedule" className="mt-6 inline-flex items-center text-indigo-600 hover:text-indigo-500">
                Schedule Session <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="py-16 md:py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-bold text-gray-900">What Our Users Say</h2>
            <p className="mt-4 text-xl text-gray-600 max-w-3xl mx-auto">
              Thousands of professionals have improved their presentation skills with our platform.
            </p>
          </div>

          <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-white p-8 rounded-lg shadow border border-gray-100">
              <div className="flex items-center">
                <img
                  src="https://images.unsplash.com/photo-1494790108377-be9c29b29330?ixlib=rb-1.2.1&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80"
                  alt="Sarah Johnson"
                  className="h-12 w-12 rounded-full"
                />
                <div className="ml-4">
                  <h4 className="text-lg font-semibold">Sarah Johnson</h4>
                  <p className="text-gray-600">Marketing Director</p>
                </div>
              </div>
              <p className="mt-6 text-gray-600">
                "The AI feedback was eye-opening. I didn't realize how many filler words I was using until I saw the analysis. After just three weeks of practice, my presentations are much more polished."
              </p>
            </div>

            <div className="bg-white p-8 rounded-lg shadow border border-gray-100">
              <div className="flex items-center">
                <img
                  src="https://images.unsplash.com/photo-1500648767791-00dcc994a43e?ixlib=rb-1.2.1&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80"
                  alt="Michael Chen"
                  className="h-12 w-12 rounded-full"
                />
                <div className="ml-4">
                  <h4 className="text-lg font-semibold">Michael Chen</h4>
                  <p className="text-gray-600">Startup Founder</p>
                </div>
              </div>
              <p className="mt-6 text-gray-600">
                "My 1-on-1 session with Simon Sinek transformed how I pitch my company. His insights on storytelling helped me secure our Series A funding. This platform is worth every penny."
              </p>
            </div>

            <div className="bg-white p-8 rounded-lg shadow border border-gray-100">
              <div className="flex items-center">
                <img
                  src="https://images.unsplash.com/photo-1438761681033-6461ffad8d80?ixlib=rb-1.2.1&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80"
                  alt="Jessica Williams"
                  className="h-12 w-12 rounded-full"
                />
                <div className="ml-4">
                  <h4 className="text-lg font-semibold">Jessica Williams</h4>
                  <p className="text-gray-600">Sales Executive</p>
                </div>
              </div>
              <p className="mt-6 text-gray-600">
                "I used to dread public speaking. The detailed feedback on my body language and voice modulation helped me identify specific areas to improve. Now I actually look forward to presentations!"
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-indigo-700 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-bold">Ready to Transform Your Presentation Skills?</h2>
            <p className="mt-4 text-xl text-indigo-100 max-w-3xl mx-auto">
              Join thousands of professionals who have elevated their communication with our platform.
            </p>
            <div className="mt-8">
              <Link
                to="/record"
                className="inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-indigo-700 bg-white hover:bg-indigo-50"
              >
                Get Started Today
                <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;