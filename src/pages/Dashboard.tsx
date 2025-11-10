import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { BarChart2, Mic, Video, MessageSquare, Clock, Volume2, Activity, Award, ArrowRight, Calendar, Download, TrendingUp, TrendingDown } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, LineChart, Line, Legend, Area, AreaChart } from 'recharts';

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

const detailedProgressData = [
  { date: 'Jan 1', bodyLanguage: 60, speech: 58, overall: 59 },
  { date: 'Jan 8', bodyLanguage: 65, speech: 62, overall: 63 },
  { date: 'Jan 15', bodyLanguage: 70, speech: 66, overall: 68 },
  { date: 'Jan 22', bodyLanguage: 75, speech: 70, overall: 72 },
  { date: 'Jan 29', bodyLanguage: 78, speech: 74, overall: 76 },
  { date: 'Feb 5', bodyLanguage: 82, speech: 76, overall: 79 },
];

const sessionHistory = [
  { id: 1, date: '2024-02-05', title: 'Product Launch Pitch', score: 82, duration: '4:32' },
  { id: 2, date: '2024-01-29', title: 'Team Meeting Presentation', score: 76, duration: '5:15' },
  { id: 3, date: '2024-01-22', title: 'Sales Demo Practice', score: 72, duration: '3:45' },
  { id: 4, date: '2024-01-15', title: 'Quarterly Review', score: 68, duration: '6:20' },
];

const Dashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');
  
  const handleExportReport = () => {
    // Create a simple text report
    const report = `
PRESENTATION ANALYSIS REPORT
Generated: ${new Date().toLocaleDateString()}

OVERALL PERFORMANCE
Overall Score: 76/100
Duration: 4:32
Speaking Pace: 145 WPM
Filler Words: 59 (13.2% of total words)

PERFORMANCE METRICS
- Clarity: 78%
- Pace: 65%
- Body Language: 82%
- Engagement: 70%
- Structure: 85%

BODY LANGUAGE ANALYSIS
- Eye Contact: 86%
- Gestures: 72%
- Posture: 80%
- Facial Expressions: 75%

RECOMMENDATIONS
1. Reduce filler words ("um", "like")
2. Slow down speaking pace in technical sections
3. Maintain more consistent eye contact

For detailed analysis, visit your dashboard.
    `.trim();
    
    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `presentation-analysis-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-gray-50 min-h-screen py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8 flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Presentation Analysis Dashboard</h1>
            <p className="mt-2 text-gray-600">
              Review your latest presentation analysis and track your progress over time.
            </p>
          </div>
          <button
            onClick={handleExportReport}
            className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            <Download className="mr-2 h-4 w-4" />
            Export Report
          </button>
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
                      <AreaChart
                        data={detailedProgressData}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis domain={[0, 100]} />
                        <Tooltip />
                        <Legend />
                        <Area type="monotone" dataKey="bodyLanguage" stackId="1" stroke="#10B981" fill="#10B981" fillOpacity={0.6} name="Body Language" />
                        <Area type="monotone" dataKey="speech" stackId="2" stroke="#6366F1" fill="#6366F1" fillOpacity={0.6} name="Speech Quality" />
                        <Line type="monotone" dataKey="overall" stroke="#F59E0B" strokeWidth={3} name="Overall Score" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
                
                <div className="bg-gray-50 p-6 rounded-lg mb-8">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Session History</h3>
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                      <thead className="bg-gray-100">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Title</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Duration</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Score</th>
                          <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Trend</th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-gray-200">
                        {sessionHistory.map((session, index) => {
                          const prevScore = index < sessionHistory.length - 1 ? sessionHistory[index + 1].score : session.score;
                          const scoreDiff = session.score - prevScore;
                          return (
                            <tr key={session.id} className="hover:bg-gray-50">
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{session.date}</td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{session.title}</td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{session.duration}</td>
                              <td className="px-6 py-4 whitespace-nowrap">
                                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                  session.score >= 80 ? 'bg-green-100 text-green-800' : 
                                  session.score >= 70 ? 'bg-yellow-100 text-yellow-800' : 
                                  'bg-red-100 text-red-800'
                                }`}>
                                  {session.score}/100
                                </span>
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm">
                                {scoreDiff > 0 ? (
                                  <span className="inline-flex items-center text-green-600">
                                    <TrendingUp className="h-4 w-4 mr-1" />
                                    +{scoreDiff}
                                  </span>
                                ) : scoreDiff < 0 ? (
                                  <span className="inline-flex items-center text-red-600">
                                    <TrendingDown className="h-4 w-4 mr-1" />
                                    {scoreDiff}
                                  </span>
                                ) : (
                                  <span className="text-gray-400">—</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
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

        {/* Achievements and Milestones */}
        <div className="bg-white rounded-lg shadow p-6 mb-8">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Achievements & Milestones</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 p-6 rounded-lg border-2 border-yellow-300">
              <div className="flex items-center justify-between mb-4">
                <Award className="h-10 w-10 text-yellow-600" />
                <span className="text-xs font-semibold text-yellow-700 bg-yellow-200 px-2 py-1 rounded">UNLOCKED</span>
              </div>
              <h3 className="text-lg font-bold text-yellow-900 mb-2">First Recording</h3>
              <p className="text-sm text-yellow-800">Completed your first presentation recording and analysis</p>
            </div>
            
            <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-lg border-2 border-green-300">
              <div className="flex items-center justify-between mb-4">
                <Award className="h-10 w-10 text-green-600" />
                <span className="text-xs font-semibold text-green-700 bg-green-200 px-2 py-1 rounded">UNLOCKED</span>
              </div>
              <h3 className="text-lg font-bold text-green-900 mb-2">Filler Fighter</h3>
              <p className="text-sm text-green-800">Reduced filler words by 30% from your first recording</p>
            </div>
            
            <div className="bg-gradient-to-br from-gray-50 to-gray-100 p-6 rounded-lg border-2 border-gray-300 opacity-60">
              <div className="flex items-center justify-between mb-4">
                <Award className="h-10 w-10 text-gray-400" />
                <span className="text-xs font-semibold text-gray-500 bg-gray-200 px-2 py-1 rounded">LOCKED</span>
              </div>
              <h3 className="text-lg font-bold text-gray-700 mb-2">Perfect Pace</h3>
              <p className="text-sm text-gray-600">Maintain 120-140 WPM for 5 consecutive recordings</p>
              <div className="mt-3">
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-gray-400 h-2 rounded-full" style={{ width: '40%' }}></div>
                </div>
                <p className="text-xs text-gray-500 mt-1">2/5 recordings</p>
              </div>
            </div>
          </div>
        </div>

        {/* Recommendations */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Personalized Recommendations</h2>
          
          <div className="mb-8 bg-blue-50 border-l-4 border-blue-500 p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <Activity className="h-5 w-5 text-blue-500" />
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-blue-800">Your Focus Area: Speaking Pace</h3>
                <p className="mt-2 text-sm text-blue-700">
                  Based on your recent presentations, we recommend focusing on slowing down your speaking pace. 
                  Your current average is 145 WPM, and the ideal range is 120-140 WPM for better audience comprehension.
                </p>
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div className="bg-indigo-50 p-6 rounded-lg border border-indigo-100">
              <h3 className="text-lg font-medium text-indigo-900 mb-4">Practice Exercises</h3>
              <div className="space-y-4">
                <div className="bg-white p-4 rounded-lg border border-indigo-200">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-semibold text-gray-900">Filler Word Reduction</h4>
                      <p className="text-sm text-gray-600 mt-1">Record yourself speaking for 2 minutes without using any filler words. Practice daily.</p>
                      <div className="mt-2 flex items-center text-xs text-indigo-600">
                        <Clock className="h-3 w-3 mr-1" />
                        <span>5 min/day</span>
                      </div>
                    </div>
                    <input type="checkbox" className="mt-1 h-5 w-5 text-indigo-600 rounded" />
                  </div>
                </div>
                
                <div className="bg-white p-4 rounded-lg border border-indigo-200">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-semibold text-gray-900">Pacing Exercise</h4>
                      <p className="text-sm text-gray-600 mt-1">Read a passage aloud while using a metronome set to 120 BPM to maintain a steady pace.</p>
                      <div className="mt-2 flex items-center text-xs text-indigo-600">
                        <Clock className="h-3 w-3 mr-1" />
                        <span>10 min/day</span>
                      </div>
                    </div>
                    <input type="checkbox" className="mt-1 h-5 w-5 text-indigo-600 rounded" />
                  </div>
                </div>
                
                <div className="bg-white p-4 rounded-lg border border-indigo-200">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-semibold text-gray-900">Gesture Practice</h4>
                      <p className="text-sm text-gray-600 mt-1">Record a 3-minute presentation focusing on purposeful hand gestures to emphasize key points.</p>
                      <div className="mt-2 flex items-center text-xs text-indigo-600">
                        <Clock className="h-3 w-3 mr-1" />
                        <span>15 min/session</span>
                      </div>
                    </div>
                    <input type="checkbox" className="mt-1 h-5 w-5 text-indigo-600 rounded" />
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-indigo-50 p-6 rounded-lg border border-indigo-100">
              <h3 className="text-lg font-medium text-indigo-900 mb-4">Suggested Resources</h3>
              <div className="space-y-4">
                <Link to="/masterclass" className="block bg-white p-4 rounded-lg border border-indigo-200 hover:border-indigo-400 transition-colors">
                  <div className="flex items-start">
                    <Video className="h-5 w-5 text-indigo-600 mt-0.5 mr-3" />
                    <div>
                      <h4 className="font-semibold text-gray-900">Vocal Variety and Emphasis</h4>
                      <p className="text-sm text-gray-600 mt-1">Masterclass by Tony Robbins</p>
                      <span className="inline-block mt-2 text-xs text-indigo-600 font-medium">45 min • Beginner</span>
                    </div>
                  </div>
                </Link>
                
                <Link to="/masterclass" className="block bg-white p-4 rounded-lg border border-indigo-200 hover:border-indigo-400 transition-colors">
                  <div className="flex items-start">
                    <BarChart2 className="h-5 w-5 text-indigo-600 mt-0.5 mr-3" />
                    <div>
                      <h4 className="font-semibold text-gray-900">The Power of Pausing</h4>
                      <p className="text-sm text-gray-600 mt-1">Article in resource library</p>
                      <span className="inline-block mt-2 text-xs text-indigo-600 font-medium">10 min read</span>
                    </div>
                  </div>
                </Link>
                
                <Link to="/masterclass" className="block bg-white p-4 rounded-lg border border-indigo-200 hover:border-indigo-400 transition-colors">
                  <div className="flex items-start">
                    <Activity className="h-5 w-5 text-indigo-600 mt-0.5 mr-3" />
                    <div>
                      <h4 className="font-semibold text-gray-900">Body Language Mastery</h4>
                      <p className="text-sm text-gray-600 mt-1">Video tutorial by Vinh Giang</p>
                      <span className="inline-block mt-2 text-xs text-indigo-600 font-medium">30 min • Intermediate</span>
                    </div>
                  </div>
                </Link>
              </div>
              <div className="mt-4">
                <Link
                  to="/masterclass"
                  className="inline-flex items-center text-indigo-600 hover:text-indigo-500 font-medium"
                >
                  Browse All Resources <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-purple-50 p-6 rounded-lg border border-purple-100">
              <h3 className="text-lg font-medium text-purple-900 mb-4">Weekly Goals</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">Complete 3 practice recordings</span>
                  <span className="text-sm font-semibold text-purple-600">2/3</span>
                </div>
                <div className="w-full bg-purple-200 rounded-full h-2">
                  <div className="bg-purple-600 h-2 rounded-full" style={{ width: '66%' }}></div>
                </div>
                
                <div className="flex items-center justify-between mt-4">
                  <span className="text-sm text-gray-700">Reduce filler words by 10%</span>
                  <span className="text-sm font-semibold text-green-600">✓ Done</span>
                </div>
                <div className="w-full bg-green-200 rounded-full h-2">
                  <div className="bg-green-600 h-2 rounded-full" style={{ width: '100%' }}></div>
                </div>
                
                <div className="flex items-center justify-between mt-4">
                  <span className="text-sm text-gray-700">Watch 2 masterclass videos</span>
                  <span className="text-sm font-semibold text-purple-600">1/2</span>
                </div>
                <div className="w-full bg-purple-200 rounded-full h-2">
                  <div className="bg-purple-600 h-2 rounded-full" style={{ width: '50%' }}></div>
                </div>
              </div>
            </div>
            
            <div className="bg-green-50 p-6 rounded-lg border border-green-100">
              <div className="flex items-start">
                <div className="flex-shrink-0">
                  <Calendar className="h-6 w-6 text-green-600" />
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-medium text-green-900">Ready for Expert Coaching?</h3>
                  <p className="mt-2 text-sm text-gray-700">
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
    </div>
  );
};

export default Dashboard;