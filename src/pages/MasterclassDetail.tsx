import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Play, Clock, Star, BookOpen, Download, Share2, Calendar, ArrowLeft, CheckCircle, MessageSquare } from 'lucide-react';

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
    reviews: 87,
    description: "Learn how to craft compelling stories that captivate your audience and make your message memorable.",
    longDescription: "In this masterclass, Simon Sinek shares his proven storytelling framework that has helped leaders around the world deliver impactful presentations. You'll learn how to identify and craft stories that resonate with your audience, create emotional connections, and make your key messages stick. Simon breaks down the science of storytelling and provides practical exercises to help you implement these techniques in your next presentation.",
    videoUrl: "#",
    resources: [
      { name: "Storytelling Framework Template", type: "PDF" },
      { name: "Story Structure Worksheet", type: "PDF" },
      { name: "Example Stories for Different Contexts", type: "PDF" }
    ],
    modules: [
      { title: "Introduction to Storytelling", duration: "8 minutes" },
      { title: "Finding Your Core Stories", duration: "12 minutes" },
      { title: "Crafting a Narrative Arc", duration: "15 minutes" },
      { title: "Delivery Techniques", duration: "10 minutes" }
    ],
    relatedMasterclasses: [2, 5]
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
    reviews: 124,
    description: "Discover how to use your body language to enhance your message and connect with your audience.",
    longDescription: "Vinh Giang combines his background in magic with communication science to reveal the secrets of powerful body language. This masterclass covers everything from stance and movement to hand gestures and facial expressions. You'll learn how to project confidence, establish credibility, and create a magnetic presence that draws your audience in. Vinh provides practical demonstrations and exercises that you can immediately apply to your presentations.",
    videoUrl: "#",
    resources: [
      { name: "Body Language Cheat Sheet", type: "PDF" },
      { name: "Gesture Library Reference Guide", type: "PDF" },
      { name: "Presence Assessment Tool", type: "Interactive" }
    ],
    modules: [
      { title: "The Psychology of Presence", duration: "12 minutes" },
      { title: "Stance and Movement", duration: "15 minutes" },
      { title: "Hand Gestures for Impact", duration: "18 minutes" },
      { title: "Facial Expressions and Eye Contact", duration: "15 minutes" }
    ],
    relatedMasterclasses: [3, 4]
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
    reviews: 156,
    description: "Master the techniques to develop a powerful, confident voice that commands attention and respect.",
    longDescription: "Tony Robbins shares his proven techniques for developing a commanding vocal presence that captivates audiences of any size. This masterclass covers vocal variety, pacing, emphasis, and the strategic use of silence. You'll learn how to modulate your voice to create emotional impact, maintain audience engagement, and deliver your message with authority. Tony provides practical exercises to strengthen your vocal instrument and expand your range as a speaker.",
    videoUrl: "#",
    resources: [
      { name: "Vocal Warm-up Routine", type: "Audio" },
      { name: "Voice Modulation Exercises", type: "PDF" },
      { name: "Emphasis Techniques Guide", type: "PDF" }
    ],
    modules: [
      { title: "Understanding Your Vocal Instrument", duration: "10 minutes" },
      { title: "Breath Control and Projection", duration: "15 minutes" },
      { title: "Vocal Variety and Emphasis", duration: "18 minutes" },
      { title: "The Strategic Use of Silence", duration: "12 minutes" }
    ],
    relatedMasterclasses: [2, 6]
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
    reviews: 98,
    description: "Learn practical techniques to manage nervousness and present with confidence, even under pressure.",
    longDescription: "In this supportive masterclass, Vinh Giang addresses the number one fear for many people: public speaking. Drawing from psychology and his experience as a performer, Vinh provides practical techniques to transform anxiety into positive energy. You'll learn pre-presentation rituals, mindset shifts, and in-the-moment recovery techniques that will help you present with confidence in any situation. This masterclass includes guided exercises and real-world examples of overcoming presentation anxiety.",
    videoUrl: "#",
    resources: [
      { name: "Anxiety Management Techniques", type: "PDF" },
      { name: "Pre-Presentation Ritual Guide", type: "PDF" },
      { name: "Confidence Building Exercises", type: "Audio" }
    ],
    modules: [
      { title: "Understanding Presentation Anxiety", duration: "8 minutes" },
      { title: "Preparation Strategies", duration: "10 minutes" },
      { title: "Pre-Presentation Rituals", duration: "12 minutes" },
      { title: "In-the-Moment Recovery Techniques", duration: "10 minutes" }
    ],
    relatedMasterclasses: [2, 3]
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
    reviews: 76,
    description: "Design visual aids that enhance your message without overwhelming your audience.",
    longDescription: "Simon Sinek reveals his approach to creating visual aids that support rather than distract from your message. This masterclass covers slide design principles, the effective use of images, data visualization, and the balance between visual and verbal elements. You'll learn how to create slides that amplify your key points, maintain audience engagement, and help your message stick. Simon provides before-and-after examples and practical templates you can adapt for your own presentations.",
    videoUrl: "#",
    resources: [
      { name: "Slide Design Templates", type: "PowerPoint/Keynote" },
      { name: "Visual Hierarchy Guide", type: "PDF" },
      { name: "Data Visualization Best Practices", type: "PDF" }
    ],
    modules: [
      { title: "Visual Design Principles", duration: "12 minutes" },
      { title: "Text and Typography", duration: "10 minutes" },
      { title: "Effective Use of Images", duration: "15 minutes" },
      { title: "Data Visualization", duration: "13 minutes" }
    ],
    relatedMasterclasses: [1, 6]
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
    reviews: 112,
    description: "Master the art of persuasive speaking to influence decisions and inspire action.",
    longDescription: "Tony Robbins reveals the psychology of influence and the art of persuasive communication in this advanced masterclass. You'll learn how to structure arguments, address objections, use emotional and logical appeals, and create compelling calls to action. Tony breaks down the elements of persuasion used by the world's most influential speakers and provides a framework for applying these techniques ethically in your own presentations. This masterclass includes case studies and practical exercises to help you implement these strategies.",
    videoUrl: "#",
    resources: [
      { name: "Persuasion Framework Template", type: "PDF" },
      { name: "Objection Handling Guide", type: "PDF" },
      { name: "Call to Action Examples", type: "PDF" }
    ],
    modules: [
      { title: "The Psychology of Influence", duration: "15 minutes" },
      { title: "Structuring Persuasive Arguments", duration: "18 minutes" },
      { title: "Emotional vs. Logical Appeals", duration: "17 minutes" },
      { title: "Creating Compelling Calls to Action", duration: "15 minutes" }
    ],
    relatedMasterclasses: [1, 3]
  }
];

const MasterclassDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [masterclass, setMasterclass] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [relatedClasses, setRelatedClasses] = useState<any[]>([]);
  const [isVideoPlaying, setIsVideoPlaying] = useState(false);

  useEffect(() => {
    // Find the masterclass by ID
    const foundMasterclass = masterclasses.find(mc => mc.id.toString() === id);
    
    if (foundMasterclass) {
      setMasterclass(foundMasterclass);
      
      // Get related masterclasses
      if (foundMasterclass.relatedMasterclasses) {
        const related = masterclasses.filter(mc => 
          foundMasterclass.relatedMasterclasses.includes(mc.id)
        );
        setRelatedClasses(related);
      }
    }
  }, [id]);

  if (!masterclass) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <BookOpen className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-lg font-medium text-gray-900">Masterclass not found</h3>
          <p className="mt-1 text-gray-500">The masterclass you're looking for doesn't exist or has been removed.</p>
          <Link
            to="/masterclass"
            className="mt-6 inline-flex items-center text-indigo-600 hover:text-indigo-500"
          >
            <ArrowLeft className="mr-2 h-5 w-5" />
            Back to Masterclasses
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-50 min-h-screen py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-6">
          <Link
            to="/masterclass"
            className="inline-flex items-center text-indigo-600 hover:text-indigo-500"
          >
            <ArrowLeft className="mr-2 h-5 w-5" />
            Back to Masterclasses
          </Link>
        </div>

        {/* Video Player Section */}
        <div className="bg-black rounded-xl overflow-hidden shadow-xl mb-8">
          <div className="relative aspect-video">
            {!isVideoPlaying ? (
              <div className="absolute inset-0">
                <img
                  src={masterclass.image}
                  alt={masterclass.title}
                  className="w-full h-full object-cover"
                />
                <div className="absolute inset-0 bg-black bg-opacity-40 flex items-center justify-center">
                  <button
                    onClick={() => setIsVideoPlaying(true)}
                    className="bg-indigo-600 hover:bg-indigo-700 text-white rounded-full p-4 transition-transform transform hover:scale-110"
                  >
                    <Play className="h-12 w-12" />
                  </button>
                </div>
              </div>
            ) : (
              <div className="w-full h-full bg-black flex items-center justify-center">
                <div className="text-center text-white">
                  <Play className="h-16 w-16 mx-auto mb-4" />
                  <p className="text-xl">Video would play here in a real implementation</p>
                  <button
                    onClick={() => setIsVideoPlaying(false)}
                    className="mt-4 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-md"
                  >
                    Return to Thumbnail
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Masterclass Info */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              <div className="p-6 border-b border-gray-200">
                <div className="flex flex-wrap items-center gap-2 text-sm text-gray-500 mb-2">
                  <span className="bg-indigo-100 text-indigo-800 px-2 py-1 rounded-full font-medium">
                    {masterclass.category}
                  </span>
                  <span>•</span>
                  <div className="flex items-center">
                    <Clock className="h-4 w-4 mr-1" />
                    <span>{masterclass.duration}</span>
                  </div>
                  <span>•</span>
                  <span>{masterclass.level}</span>
                </div>
                
                <h1 className="text-2xl md:text-3xl font-bold text-gray-900 mb-2">
                  {masterclass.title}
                </h1>
                
                <div className="flex items-center mb-4">
                  <span className="text-gray-700 font-medium mr-1">By</span>
                  <span className="text-indigo-600 font-medium">{masterclass.instructor}</span>
                </div>
                
                <div className="flex items-center">
                  <div className="flex items-center">
                    <Star className="h-5 w-5 text-yellow-400" />
                    <span className="ml-1 text-gray-700 font-medium">{masterclass.rating}</span>
                  </div>
                  <span className="mx-2 text-gray-400">|</span>
                  <span className="text-gray-500">{masterclass.reviews} reviews</span>
                </div>
              </div>
              
              {/* Tabs */}
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
                    onClick={() => setActiveTab('modules')}
                    className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                      activeTab === 'modules'
                        ? 'border-indigo-500 text-indigo-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    Modules
                  </button>
                  <button
                    onClick={() => setActiveTab('resources')}
                    className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                      activeTab === 'resources'
                        ? 'border-indigo-500 text-indigo-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    Resources
                  </button>
                </nav>
              </div>
              
              {/* Tab Content */}
              <div className="p-6">
                {activeTab === 'overview' && (
                  <div>
                    <h2 className="text-xl font-semibold text-gray-900 mb-4">About This Masterclass</h2>
                    <p className="text-gray-700 mb-6">
                      {masterclass.longDescription}
                    </p>
                    
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">What You'll Learn</h3>
                    <ul className="space-y-2 mb-6">
                      {masterclass.modules.map((module: any, index: number) => (
                        <li key={index} className="flex items-start">
                          <CheckCircle className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                          <span className="text-gray-700">{module.title}</span>
                        </li>
                      ))}
                    </ul>
                    
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">Who This Masterclass Is For</h3>
                    <ul className="space-y-2">
                      <li className="flex items-start">
                        <CheckCircle className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                        <span className="text-gray-700">Professionals who want to improve their presentation skills</span>
                      </li>
                      <li className="flex items-start">
                        <CheckCircle className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                        <span className="text-gray-700">Leaders who need to communicate effectively with their teams</span>
                      </li>
                      <li className="flex items-start">
                        <CheckCircle className="h-5 w-5 text-green-500 mr-2 flex-shrink-0 mt-0.5" />
                        <span className="text-gray-700">Anyone who wants to become a more confident and impactful speaker</span>
                      </li>
                    </ul>
                  </div>
                )}
                
                {activeTab === 'modules' && (
                  <div>
                    <h2 className="text-xl font-semibold text-gray-900 mb-4">Course Content</h2>
                    <div className="space-y-4">
                      {masterclass.modules.map((module: any, index: number) => (
                        <div key={index} className="bg-gray-50 p-4 rounded-lg">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center">
                              <div className="bg-indigo-100 text-indigo-800 w-8 h-8 rounded-full flex items-center justify-center font-semibold mr-3">
                                {index + 1}
                              </div>
                              <h3 className="font-medium text-gray-900">{module.title}</h3>
                            </div>
                            <div className="flex items-center">
                              <Clock className="h-4 w-4 text-gray-400 mr-1" />
                              <span className="text-sm text-gray-500">{module.duration}</span>
                            </div>
                          </div>
                          {index === 0 && (
                            <div className="mt-3 ml-11">
                              <button className="inline-flex items-center text-indigo-600 hover:text-indigo-500 text-sm font-medium">
                                <Play className="h-4 w-4 mr-1" />
                                Preview this module
                              </button>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {activeTab === 'resources' && (
                  <div>
                    <h2 className="text-xl font-semibold text-gray-900 mb-4">Downloadable Resources</h2>
                    <div className="space-y-4">
                      {masterclass.resources.map((resource: any, index: number) => (
                        <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                          <div className="flex items-center">
                            <div className="bg-indigo-100 p-2 rounded-md mr-3">
                              <Download className="h-5 w-5 text-indigo-600" />
                            </div>
                            <div>
                              <h3 className="font-medium text-gray-900">{resource.name}</h3>
                              <p className="text-sm text-gray-500">{resource.type}</p>
                            </div>
                          </div>
                          <button className="px-3 py-1 bg-indigo-100 text-indigo-700 rounded-md text-sm font-medium hover:bg-indigo-200">
                            Download
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            {/* Reviews Section */}
            <div className="mt-8 bg-white rounded-lg shadow-md overflow-hidden">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-xl font-semibold text-gray-900">Student Reviews</h2>
              </div>
              
              <div className="p-6">
                <div className="flex items-center mb-6">
                  <div className="flex items-center mr-4">
                    <Star className="h-6 w-6 text-yellow-400" />
                    <span className="ml-2 text-2xl font-bold text-gray-900">{masterclass.rating}</span>
                  </div>
                  <div className="text-sm text-gray-500">
                    Based on {masterclass.reviews} reviews
                  </div>
                </div>
                
                <div className="space-y-6">
                  <div className="border-b border-gray-200 pb-6">
                    <div className="flex items-start">
                      <img
                        src="https://images.unsplash.com/photo-1494790108377-be9c29b29330?ixlib=rb-1.2.1&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80"
                        alt="Reviewer"
                        className="h-10 w-10 rounded-full mr-4"
                      />
                      <div>
                        <div className="flex items-center mb-1">
                          <h3 className="font-medium text-gray-900 mr-2">Sarah Johnson</h3>
                          <div className="flex">
                            {[...Array(5)].map((_, i) => (
                              <Star key={i} className="h-4 w-4 text-yellow-400" />
                            ))}
                          </div>
                        </div>
                        <p className="text-sm text-gray-500 mb-2">2 weeks ago</p>
                        <p className="text-gray-700">
                          This masterclass completely transformed my approach to presentations. The techniques are practical and easy to implement. I've already received positive feedback on my improved delivery style.
                        </p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="border-b border-gray-200 pb-6">
                    <div className="flex items-start">
                      <img
                        src="https://images.unsplash.com/photo-1500648767791-00dcc994a43e?ixlib=rb-1.2.1&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80"
                        alt="Reviewer"
                        className="h-10 w-10 rounded-full mr-4"
                      />
                      <div>
                        <div className="flex items-center mb-1">
                          <h3 className="font-medium text-gray-900 mr-2">Michael Chen</h3>
                          <div className="flex">
                            {[...Array(4)].map((_, i) => (
                              <Star key={i} className="h-4 w-4 text-yellow-400" />
                            ))}
                            {[...Array(1)].map((_, i) => (
                              <Star key={i} className="h-4 w-4 text-gray-300" />
                            ))}
                          </div>
                        </div>
                        <p className="text-sm text-gray-500 mb-2">1 month ago</p>
                        <p className="text-gray-700">
                          Great content and very insightful. I would have appreciated more examples, but overall the techniques taught are valuable and have already helped me improve my presentations.
                        </p>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex items-start">
                      <img
                        src="https://images.unsplash.com/photo-1438761681033-6461ffad8d80?ixlib=rb-1.2.1&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80"
                        alt="Reviewer"
                        className="h-10 w-10 rounded-full mr-4"
                      />
                      <div>
                        <div className="flex items-center mb-1">
                          <h3 className="font-medium text-gray-900 mr-2">Jessica Williams</h3>
                          <div className="flex">
                            {[...Array(5)].map((_, i) => (
                              <Star key={i} className="h-4 w-4 text-yellow-400" />
                            ))}
                          </div>
                        </div>
                        <p className="text-sm text-gray-500 mb-2">2 months ago</p>
                        <p className="text-gray-700">
                          Absolutely worth every minute! The instructor breaks down complex concepts into actionable steps. I've watched it twice already and keep finding new insights each time.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="mt-6 text-center">
                  <button className="text-indigo-600 hover:text-indigo-500 font-medium">
                    View All Reviews
                  </button>
                </div>
              </div>
            </div>
          </div>
          
          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-md overflow-hidden sticky top-6">
              <div className="p-6 border-b border-gray-200">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-semibold text-gray-900">Instructor</h2>
                  <button className="text-indigo-600 hover:text-indigo-500 text-sm font-medium">
                    View Profile
                  </button>
                </div>
                
                <div className="flex items-center">
                  <img
                    src={masterclass.instructor === "Simon Sinek" 
                      ? "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80"
                      : masterclass.instructor === "Vinh Giang"
                      ? "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80"
                      : "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80"
                    }
                    alt={masterclass.instructor}
                    className="h-14 w-14 rounded-full mr-4"
                  />
                  <div>
                    <h3 className="font-medium text-gray-900">{masterclass.instructor}</h3>
                    <p className="text-sm text-gray-500">Presentation Expert</p>
                  </div>
                </div>
              </div>
              
              <div className="p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Actions</h2>
                
                <div className="space-y-4">
                  <button className="w-full flex items-center justify-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                    <BookOpen className="mr-2 h-5 w-5" />
                    Continue Learning
                  </button>
                  
                  <button className="w-full flex items-center justify-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                    <Download className="mr-2 h-5 w-5" />
                    Download All Resources
                  </button>
                  
                  <button className="w-full flex items-center justify-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                    <Share2 className="mr-2 h-5 w-5" />
                    Share
                  </button>
                </div>
                
                <div className="mt-6 pt-6 border-t border-gray-200">
                  <h2 className="text-lg font-semibold text-gray-900 mb-4">Need Personalized Help?</h2>
                  <Link
                    to="/schedule"
                    className="w-full flex items-center justify-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
                  >
                    <Calendar className="mr-2 h-5 w-5" />
                    Schedule 1-on-1 Session
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Related Masterclasses */}
        {relatedClasses.length > 0 && (
          <div className="mt-12">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Related Masterclasses</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {relatedClasses.map((related) => (
                <div key={related.id} className="bg-white rounded-lg shadow-md overflow-hidden transition-transform duration-300 hover:shadow-lg hover:-translate-y-1">
                  <div className="relative h-48">
                    <img
                      src={related.image}
                      alt={related.title}
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute top-0 right-0 bg-indigo-600 text-white text-xs font-bold px-3 py-1 m-2 rounded-full">
                      {related.category}
                    </div>
                  </div>
                  
                  <div className="p-6">
                    <div className="flex items-center text-sm text-gray-500 mb-2">
                      <Clock className="h-4 w-4 mr-1" />
                      <span>{related.duration}</span>
                      <span className="mx-2">•</span>
                      <span>{related.level}</span>
                    </div>
                    
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">{related.title}</h3>
                    <p className="text-gray-600 mb-4">{related.description}</p>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <div className="flex items-center">
                          <Star className="h-5 w-5 text-yellow-400" />
                          <span className="ml-1 text-gray-700 font-medium">{related.rating}</span>
                        </div>
                        <span className="mx-2 text-gray-300">|</span>
                        <span className="text-gray-600">By {related.instructor}</span>
                      </div>
                      
                      <Link
                        to={`/masterclass/${related.id}`}
                        className="text-indigo-600 hover:text-indigo-500 font-medium"
                      >
                        View
                      </Link>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default MasterclassDetail;