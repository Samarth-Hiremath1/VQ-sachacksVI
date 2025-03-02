import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { BarChart2, Mic, Video, MessageSquare, Clock, Volume2, Activity, Award, ArrowRight, Calendar } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, LineChart, Line, Legend } from 'recharts';

// Mock data for charts
const performanceData = [
  { name: 'Clarity', score: 78 },
  { name: 'Pace', score: 65 },
  { name: 'Body Language', score: 82 },
  { name: 'Engagement', score: 70 },
  { name: 'Structure', score: 85 },
];

const fillerWordsData = [
  { name: 'Um', count: 12 },
  { name: 'Uh', count: 8 },
  { name: 'Like', count: 15 },
  { name: 'You know', count: 6 },
  { name: 'So', count: 18 },
];

const progressData = [
  { date: 'Week 1', score: 62 },
  { date: 'Week 2', score: 68 },
  { date: 'Week 3', score: 75 },
  { date: 'Week 4', score: 82 },
];

const radarData = [
  { subject: 'Vocal Variety', A: 65, fullMark: 100 },
  { subject: 'Pacing', A: 78, fullMark: 100 },
  { subject: 'Eye Contact', A: 86, fullMark: 100 },
  { subject: 'Gestures', A: 72, fullMark: 100 },
  { subject: 'Posture', A: 80, fullMark: 100 },
  { subject: 'Confidence', A: 75, fullMark: 100 },
];

const Dashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <div className="bg-gray-50 min-h-screen py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Presentation Analysis Dashboard</h1>
          <p className="mt-2 text-gray-600">
            Review your latest presentation analysis and track your progress over time.
          </p>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-indigo-100 text-indigo-600">
                <Award className="h-8 w-8" />
              </div>
              <div className="ml-4">
                <h2 className="text-sm font-medium text-gray-500">Overall Score</h2>
                <p className="text-3xl font-semibold text-gray-900">76/100</p>
              </div>
            </div>
            <div className="mt-4">
              <div className="w-full bg-gray-200 rounded-full h-2.5">
                <div className="bg-indigo-600 h-2.5 rounded-full" style={{ width: '76%' }}></div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-green-100 text-green-600">
                <Clock className="h-8 w-8" />
              </div>
              <div className="ml-4">
                <h2 className="text-sm font-medium text-gray-500">Duration</h2>
                <p className="text-3xl font-semibold text-gray-900">4:32</p>
              </div>
            </div>
            <p className="mt-4 text-sm text-gray-600">
              Ideal length for your content type
            </p>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-yellow-100 text-yellow-600">
                <Volume2 className="h-8 w-8" />
              </div>
              <div className="ml-4">
                <h2 className="text-sm font-medium text-gray-500">Speaking Pace</h2>
                <p className="text-3xl font-semibold text-gray-900">145 WPM</p>
              </div>
            </div>
            <p className="mt-4 text-sm text-gray-600">
              Slightly faster than ideal (120-140 WPM)
            </p>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <div className="p-3 rounded-full bg-blue-100 text-blue-600">
                <MessageSquare className="h-8 w-8" />
              </div>
              <div className="ml-4">
                <h2 className="text-sm font-medium text-gray-500">Filler Words</h2>
                <p className="text-3xl font-semibold text-gray-900">59</p>
              </div>
            </div>
            <p className="mt-4 text-sm text-gray-600">
              13.2% of total words (aim for &lt;10%)
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="bg-white rounded-lg shadow mb-8">
          <div className="border-b border-gray-200">
            <nav className="flex -mb-px">
              <button
                onClick={() => setActiveTab('overview')}
                className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                  activeTab === 'overview'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Overview
              </button>
              <button
                onClick={() => setActiveTab('verbal')}
                className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                  activeTab === 'verbal'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Verbal Analysis
              </button>
              <button
                onClick={() => setActiveTab('nonverbal')}
                className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                  activeTab === 'nonverbal'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Non-verbal Analysis
              </button>
              <button
                onClick={() => setActiveTab('progress')}
                className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                  activeTab === 'progress'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Progress
              </button>
            </nav>
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {activeTab === 'overview' && (
              <div>
                <h2 className="text-xl font-semibold text-gray-900 mb-6">Performance Overview</h2>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div className="bg-gray-50 p-6 rounded-lg">
                    <h3 className="text-lg font-medium text-gray-900 mb-4">Performance Metrics</h3>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={performanceData}
                          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" />
                          <YAxis domain={[0, 100]} />
                          <Tooltip />
                          <Bar dataKey="score" fill="#6366F1" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 p-6 rounded-lg">
                    <h3 className="text-lg font-medium text-gray-900 mb-4">Presentation Skills</h3>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <RadarChart outerRadius={90} data={radarData}>
                          <PolarGrid />
                          <PolarAngleAxis dataKey="subject" />
                          <PolarRadiusAxis domain={[0, 100]} />
                          <Radar name="Skills" dataKey="A" stroke="#6366F1" fill="#6366F1" fillOpacity={0.6} />
                        </RadarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
                
                <div className="mt-8 bg-gray-50 p-6 rounded-lg">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Key Insights</h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                      <h4 className="font-medium text-gray-900">Strengths</h4>
                      <ul className="mt-2 space-y-2">
                        <li className="flex items-start">
                          <span className="flex-shrink-0 h-5 w-5 text-green-500">✓</span>
                          <span className="ml-2 text-gray-600">Strong opening that captured attention</span>
                        </li>
                        <li className="flex items-start">
                          <span className="flex-shrink-0 h-5 w-5 text-green-500">✓</span>
                          <span className="ml-2 text-gray-600">Clear structure with logical flow</span>
                        </li>
                        <li className="flex items-start">
                          <span className="flex-shrink-0 h-5 w-5 text-green-500">✓</span>
                          <span className="ml-2 text-gray-600">Effective use of gestures to emphasize points</span>
                        </li>
                      </ul>
                    </div>
                    
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                      <h4 className="font-medium text-gray-900">Areas for Improvement</h4>
                      <ul className="mt-2 space-y-2">
                        <li className="flex items-start">
                          <span className="flex-shrink-0 h-5 w-5 text-red-500">✗</span>
                          <span className="ml-2 text-gray-600">Reduce filler words ("um", "like")</span>
                        </li>
                        <li className="flex items-start">
                          <span className="flex-shrink-0 h-5 w-5 text-red-500">✗</span>
                          <span className="ml-2 text-gray-600">Slow down speaking pace in technical sections</span>
                        </li>
                        <li className="flex items-start">
                          <span className="flex-shrink-0 h-5 w-5 text-red-500">✗</span>
                          <span className="ml-2 text-gray-600">Maintain more consistent eye contact</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {activeTab === 'verbal' && (
              <div>
                <h2 className="text-xl font-semibold text-gray-900 mb-6">Verbal Communication Analysis</h2>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div className="bg-gray-50 p-6 rounded-lg">
                    <h3 className="text-lg font-medium text-gray-900 mb-4">Filler Words</h3>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={fillerWordsData}
                          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" />
                          <YAxis />
                          <Tooltip />
                          <Bar dataKey="count" fill="#6366F1" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 p-6 rounded-lg">
                    <h3 className="text-lg font-medium text-gray-900 mb-4">Voice Analysis</h3>
                    <div className="space-y-6">
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium text-gray-700">Volume</span>
                          <span className="text-sm font-medium text-gray-700">85%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div className="bg-indigo-600 h-2.5 rounded-full" style={{ width: '85%' }}></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium text-gray-700">Clarity</span>
                          <span className="text-sm font-medium text-gray-700">78%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div className="bg-indigo-600 h-2.5 rounded-full" style={{ width: '78%' }}></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium text-gray-700">Vocal Variety</span>
                          <span className="text-sm font-medium text-gray-700">65%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div className="bg-indigo-600 h-2.5 rounded-full" style={{ width: '65%' }}></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium text-gray-700">Pace</span>
                          <span className="text-sm font-medium text-gray-700">70%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div className="bg-indigo-600 h-2.5 rounded-full" style={{ width: '70%' }}></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="mt-8 bg-gray-50 p-6 rounded-lg">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Content Analysis</h3>
                  
                  <div className="space-y-6">
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                      <h4 className="font-medium text-gray-900">Key Message Clarity</h4>
                      <p className="mt-2 text-gray-600">
                        Your main message was clear and repeated effectively throughout the presentation. 
                        The audience would easily understand your core point about "improving team collaboration through better communication tools."
                      </p>
                    </div>
                    
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                      <h4 className="font-medium text-gray-900">Structure Analysis</h4>
                      <p className="mt-2 text-gray-600">
                        Your presentation had a strong opening and conclusion, but the middle section could be more organized. 
                        Consider using clearer transitions between points to help the audience follow your logic.
                      </p>
                    </div>
                    
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                      <h4 className="font-medium text-gray-900">Language Effectiveness</h4>
                      <p className="mt-2 text-gray-600">
                        You used appropriate technical terms for your audience. However, some analogies could be simplified 
                        to make complex concepts more accessible. Your storytelling was engaging and relevant.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {activeTab === 'nonverbal' && (
              <div>
                <h2 className="text-xl font-semibold text-gray-900 mb-6">Non-verbal Communication Analysis</h2>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div className="bg-gray-50 p-6 rounded-lg">
                    <h3 className="text-lg font-medium text-gray-900 mb-4">Body Language</h3>
                    <div className="space-y-6">
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium text-gray-700">Eye Contact</span>
                          <span className="text-sm font-medium text-gray-700">86%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div className="bg-indigo-600 h-2.5 rounded-full" style={{ width: '86%' }}></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium text-gray-700">Gestures</span>
                          <span className="text-sm font-medium text-gray-700">72%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div className="bg-indigo-600 h-2.5 rounded-full" style={{ width: '72%' }}></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium text-gray-700">Posture</span>
                          <span className="text-sm font-medium text-gray-700">80%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div className="bg-indigo-600 h-2.5 rounded-full" style={{ width: '80%' }}></div>
                        </div>
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium text-gray-700">Facial Expressions</span>
                          <span className="text-sm font-medium text-gray-700">75%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2.5">
                          <div className="bg-indigo-600 h-2.5 rounded-full" style={{ width: '75%' }}></div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 p-6 rounded-lg">
                    <h3 className="text-lg font-medium text-gray-900 mb-4">Movement Analysis</h3>
                    <div className="space-y-4">
                      <div className="bg-white p-4 rounded-lg border border-gray-200">
                        <h4 className="font-medium text-gray-900">Stage Presence</h4>
                        <p className="mt-2 text-gray-600">
                          You utilized the space well, moving purposefully to emphasize key points. 
                          Your movements were natural and not distracting.
                        </p>
                      </div>
                      
                      <div className="bg-white p-4 rounded-lg border border-gray-200">
                        <h4 className="font-medium text-gray-900">Distracting Movements</h4>
                        <p className="mt-2 text-gray-600">
                          Occasional fidgeting with hands when not gesturing. 
                          Try to maintain a more relaxed posture when not actively gesturing.
                        </p>
                      </div>
                      
                      <div className="bg-white p-4 rounded-lg border border-gray-200">
                        <h4 className="font-medium text-gray-900">Energy Level</h4>
                        <p className="mt-2 text-gray-600">
                          Your energy was appropriate for the content, with good variation to emphasize important points.
                          Consider increasing energy slightly during the call-to-action section.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="mt-8 bg-gray-50 p-6 rounded-lg">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Visual Aids Usage</h3>
                  
                  <div className="space-y-6">
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                      <h4 className="font-medium text-gray-900">Slide Interaction</h4>
                      <p className="mt-2 text-gray-600">
                        You maintained good balance between looking at slides and audience. 
                        Avoided the common mistake of reading directly from slides.
                      </p>
                    </div>
                    
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                      <h4 className="font-medium text-gray-900">Pointing and Referencing</h4>
                      <p className="mt-2 text-gray-600">
                        Effective use of gestures to highlight key data points on slides.
                        Consider using more specific pointing to guide audience attention to complex visuals.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {activeTab === 'progress' && (
              <div>
                <h2 className="text-xl font-semibold text-gray-900 mb-6">Progress Tracking</h2>
                
                <div className="bg-gray-50 p-6 rounded-lg mb-8">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Performance Over Time</h3>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={progressData}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis domain={[0, 100]} />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="score" stroke="#6366F1" activeDot={{ r: 8 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                
                <div className="bg-gray-50 p-6 rounded-lg">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Improvement Areas</h3>
                  
                  <div className="space-y-6">
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                      <div className="flex justify-between items-center">
                        <h4 className="font-medium text-gray-900">Filler Words</h4>
                        <span className="text-green-600 font-medium">↓ 32% decrease</span>
                      </div>
                      <p className="mt-2 text-gray-600">
                        You've significantly reduced filler words from 87 to 59 over the past month.
                        Continue practicing pausing instead of using fillers.
                      </p>
                    </div>
                    
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                      <div className="flex justify-between items-center">
                        <h4 className="font-medium text-gray-900">Speaking Pace</h4>
                        <span className="text-yellow-600 font-medium">↓ 8% decrease</span>
                      </div>
                      <p className="mt-2 text-gray-600">
                        Your speaking pace has improved from 158 WPM to 145 WPM.
                        Still slightly above the ideal range of 120-140 WPM.
                      </p>
                    </div>
                    
                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                      <div className="flex justify-between items-center">
                        <h4 className="font-medium text-gray-900">Body Language</h4>
                        <span className="text-green-600 font-medium">↑ 15% increase</span>
                      </div>
                      <p className="mt-2 text-gray-600">
                        Your gestures and posture have become more natural and confident.
                        Eye contact has improved significantly.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Recommendations */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Personalized Recommendations</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-indigo-50 p-6 rounded-lg border border-indigo-100">
              <h3 className="text-lg font-medium text-indigo-900 mb-4">Practice Exercises</h3>
              <ul className="space-y-3">
                <li className="flex items-start">
                  <div className="flex-shrink-0 h-5 w-5 text-indigo-600">•</div>
                  <span className="ml-2 text-gray-700">
                    <strong>Filler Word Reduction:</strong> Record yourself speaking for 2 minutes without using any filler words. Practice daily.
                  </span>
                </li>
                <li className="flex items-start">
                  <div className="flex-shrink-0 h-5 w-5 text-indigo-600">•</div>
                  <span className="ml-2 text-gray-700">
                    <strong>Pacing Exercise:</strong> Read a passage aloud while using a metronome set to 120 BPM to maintain a steady pace.
                  </span>
                </li>
                <li className="flex items-start">
                  <div className="flex-shrink-0 h-5 w-5 text-indigo-600">•</div>
                  <span className="ml-2 text-gray-700">
                    <strong>Gesture Practice:</strong> Record a 3-minute presentation focusing on purposeful hand gestures to emphasize key points.
                  </span>
                </li>
              </ul>
            </div>
            
            <div className="bg-indigo-50 p-6 rounded-lg border border-indigo-100">
              <h3 className="text-lg font-medium text-indigo-900 mb-4">Suggested Resources</h3>
              <ul className="space-y-3">
                <li className="flex items-start">
                  <div className="flex-shrink-0 h-5 w-5 text-indigo-600">•</div>
                  <span className="ml-2 text-gray-700">
                    <strong>Masterclass:</strong> "Vocal Variety and Emphasis" by Tony Robbins
                  </span>
                </li>
                <li className="flex items-start">
                  <div className="flex-shrink-0 h-5 w-5 text-indigo-600">•</div>
                  <span className="ml-2 text-gray-700">
                    <strong>Article:</strong> "The Power of Pausing" in our resource library
                  </span>
                </li>
                <li className="flex items-start">
                  <div className="flex-shrink-0 h-5 w-5 text-indigo-600">•</div>
                  <span className="ml-2 text-gray-700">
                    <strong>Video Tutorial:</strong> "Body Language Mastery" by Vinh Giang
                  </span>
                </li>
              </ul>
              <div className="mt-4">
                <Link
                  to="/masterclass"
                  className="inline-flex items-center text-indigo-600 hover:text-indigo-500"
                >
                  Browse All Resources <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </div>
            </div>
          </div>
          
          <div className="mt-8 bg-green-50 p-6 rounded-lg border border-green-100">
            <div className="flex items-start">
              <div className="flex-shrink-0">
                <Calendar className="h-6 w-6 text-green-600" />
              </div>
              <div className="ml-4">
                <h3 className="text-lg font-medium text-green-900">Ready for Expert Coaching?</h3>
                <p className="mt-2 text-gray-700">
                  Schedule a 1-on-1 session with one of our presentation experts to get personalized feedback and coaching.
                </p>
                <div className="mt-4">
                  <Link
                    to="/schedule"
                    className="inline-flex items-center justify-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                  >
                    Schedule a Session
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;