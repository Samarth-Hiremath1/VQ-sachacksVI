import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Link } from 'react-router-dom';
import Webcam from 'react-webcam';
import { Mic, Video, StopCircle, Play, Save, Trash2, Info, CheckCircle, ArrowRight, BarChart2 } from 'lucide-react';
import { Pose, POSE_CONNECTIONS } from '@mediapipe/pose'; // Ensure POSE_CONNECTIONS is imported
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import * as cameraUtils from '@mediapipe/camera_utils';

const Record: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const [recording, setRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([]);
  const [recordingTime, setRecordingTime] = useState(0);
  const [recordingComplete, setRecordingComplete] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [transcript, setTranscript] = useState<string>(''); // Live transcript state
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const poseRef = useRef<Pose | null>(null);
  const cameraRef = useRef<cameraUtils.Camera | null>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  
  // Real-time feedback states
  const [postureScore, setPostureScore] = useState<number>(0);
  const [gestureQuality, setGestureQuality] = useState<string>('Neutral');
  const [speakingPace, setSpeakingPace] = useState<number>(0);
  const [volumeLevel, setVolumeLevel] = useState<number>(0);
  const [fillerWordCount, setFillerWordCount] = useState<number>(0);
  const wordCountRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);

  // Helper function to analyze posture from pose landmarks
  const analyzePosture = (landmarks: any[]) => {
    if (!landmarks || landmarks.length < 33) return 0;
    
    // Get key landmarks for posture analysis
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];
    const leftHip = landmarks[23];
    const rightHip = landmarks[24];
    const nose = landmarks[0];
    
    // Calculate shoulder alignment (should be level)
    const shoulderDiff = Math.abs(leftShoulder.y - rightShoulder.y);
    const shoulderScore = Math.max(0, 100 - (shoulderDiff * 500));
    
    // Calculate spine alignment (shoulders should be above hips)
    const avgShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
    const avgHipY = (leftHip.y + rightHip.y) / 2;
    const spineAlignment = avgShoulderY < avgHipY ? 100 : 50;
    
    // Calculate head position (nose should be roughly centered above shoulders)
    const avgShoulderX = (leftShoulder.x + rightShoulder.x) / 2;
    const headAlignment = Math.max(0, 100 - (Math.abs(nose.x - avgShoulderX) * 300));
    
    // Overall posture score
    return Math.round((shoulderScore + spineAlignment + headAlignment) / 3);
  };
  
  // Helper function to analyze gestures
  const analyzeGestures = (landmarks: any[]) => {
    if (!landmarks || landmarks.length < 33) return 'No Detection';
    
    const leftWrist = landmarks[15];
    const rightWrist = landmarks[16];
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];
    
    // Check if hands are visible and active
    const leftHandActive = leftWrist.visibility > 0.5 && leftWrist.y < leftShoulder.y;
    const rightHandActive = rightWrist.visibility > 0.5 && rightWrist.y < rightShoulder.y;
    
    if (leftHandActive && rightHandActive) return 'Active - Both Hands';
    if (leftHandActive || rightHandActive) return 'Active - One Hand';
    return 'Neutral';
  };

  // Initialize MediaPipe Pose
  useEffect(() => {
    const pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });

    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    pose.onResults((results) => {
      if (canvasRef.current && webcamRef.current?.video) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Match canvas size to video
        canvas.width = webcamRef.current.video.videoWidth;
        canvas.height = webcamRef.current.video.videoHeight;

        // Clear the canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw pose landmarks and connections if detected
        if (results.poseLandmarks) {
          drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
            color: '#00FF00', // Green lines
            lineWidth: 4,
          });
          drawLandmarks(ctx, results.poseLandmarks, {
            color: '#FF0000', // Red points
            lineWidth: 2,
            radius: 4,
          });
          
          // Analyze posture and gestures in real-time
          if (recording) {
            const posture = analyzePosture(results.poseLandmarks);
            const gesture = analyzeGestures(results.poseLandmarks);
            setPostureScore(posture);
            setGestureQuality(gesture);
          }
        }
      }
    });

    poseRef.current = pose;

    return () => {
      pose.close();
    };
  }, [recording]);

  // Initialize Web Speech API
  useEffect(() => {
    const SpeechRecognition = (window.SpeechRecognition || window.webkitSpeechRecognition) as new () => SpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onresult = (event: SpeechRecognitionEvent) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcriptPiece = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcriptPiece + ' ';
            
            // Count words for pace calculation
            const words = finalTranscript.trim().split(/\s+/).filter(w => w.length > 0);
            wordCountRef.current += words.length;
            
            // Calculate speaking pace (WPM)
            if (startTimeRef.current > 0) {
              const elapsedMinutes = (Date.now() - startTimeRef.current) / 60000;
              const wpm = Math.round(wordCountRef.current / elapsedMinutes);
              setSpeakingPace(wpm);
            }
            
            // Detect filler words
            const fillerWords = ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'literally'];
            const lowerText = finalTranscript.toLowerCase();
            let count = 0;
            fillerWords.forEach(filler => {
              const regex = new RegExp(`\\b${filler}\\b`, 'gi');
              const matches = lowerText.match(regex);
              if (matches) count += matches.length;
            });
            setFillerWordCount(prev => prev + count);
          } else {
            interimTranscript += transcriptPiece;
          }
        }
        setTranscript(prev => prev + finalTranscript + interimTranscript);
      };

      recognition.onerror = (event: Event) => {
        console.error('Speech recognition error:', event);
      };

      recognition.onend = () => {
        if (recording) recognition.start(); // Restart if still recording
      };

      recognitionRef.current = recognition;
    } else {
      console.error('SpeechRecognition API not supported in this browser.');
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, [recording]);

  // Start camera and pose detection
  useEffect(() => {
    const startCamera = async () => {
      if (webcamRef.current?.video && poseRef.current && !cameraRef.current) {
        const videoElement = webcamRef.current.video;
        const camera = new cameraUtils.Camera(videoElement, {
          onFrame: async () => {
            if (poseRef.current && videoElement) {
              await poseRef.current.send({ image: videoElement });
            }
          },
          width: 1280,
          height: 720,
        });
        await camera.start();
        cameraRef.current = camera;
      }
    };

    startCamera();

    return () => {
      if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
      }
    };
  }, []);

  const handleStartRecording = useCallback(() => {
    setRecording(true);
    setRecordedChunks([]);
    setRecordingTime(0);
    setRecordingComplete(false);
    setAnalysisComplete(false);
    setTranscript('');
    setPostureScore(0);
    setGestureQuality('Neutral');
    setSpeakingPace(0);
    setVolumeLevel(0);
    setFillerWordCount(0);
    wordCountRef.current = 0;
    startTimeRef.current = Date.now();

    if (webcamRef.current && webcamRef.current.stream) {
      mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
        mimeType: 'video/webm',
      });

      mediaRecorderRef.current.addEventListener('dataavailable', ({ data }) => {
        if (data.size > 0) {
          setRecordedChunks((prev) => [...prev, data]);
        }
      });

      mediaRecorderRef.current.start();

      timerRef.current = setInterval(() => {
        setRecordingTime((prevTime) => prevTime + 1);
      }, 1000);

      if (recognitionRef.current) {
        recognitionRef.current.start();
      }
      
      // Setup audio volume monitoring
      try {
        const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
        const audioContext = new AudioContext();
        const analyser = audioContext.createAnalyser();
        const microphone = audioContext.createMediaStreamSource(webcamRef.current.stream);
        
        analyser.smoothingTimeConstant = 0.8;
        analyser.fftSize = 1024;
        
        microphone.connect(analyser);
        
        audioContextRef.current = audioContext;
        analyserRef.current = analyser;
        
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        
        const updateVolume = () => {
          if (analyserRef.current && recording) {
            analyserRef.current.getByteFrequencyData(dataArray);
            const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
            setVolumeLevel(Math.round((average / 255) * 100));
            requestAnimationFrame(updateVolume);
          }
        };
        
        updateVolume();
      } catch (error) {
        console.error('Error setting up audio monitoring:', error);
      }
    }
  }, [recording]);

  const handleStopRecording = useCallback(() => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setRecording(false);
    setRecordingComplete(true);

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const handleAnalyze = useCallback(() => {
    setAnalyzing(true);
    setTimeout(() => {
      setAnalyzing(false);
      setAnalysisComplete(true);
    }, 3000);
  }, []);

  const handleDownload = useCallback(() => {
    if (recordedChunks.length) {
      const blob = new Blob(recordedChunks, { type: 'video/webm' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      document.body.appendChild(a);
      a.style.display = 'none';
      a.href = url;
      a.download = 'presentation-recording.webm';
      a.click();
      window.URL.revokeObjectURL(url);
    }
  }, [recordedChunks]);

  const handleReset = useCallback(() => {
    setRecordedChunks([]);
    setRecordingTime(0);
    setRecordingComplete(false);
    setAnalysisComplete(false);
    setTranscript('');
  }, []);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-gray-50 min-h-screen py-12">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h1 className="text-3xl font-bold text-gray-900">Record Your Presentation</h1>
          <p className="mt-4 text-xl text-gray-600 max-w-3xl mx-auto">
            Capture your presentation with pose tracking and live speech-to-text for AI analysis.
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-gray-800">Recording Studio</h2>
              <div className="flex items-center space-x-2">
                <span
                  className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                    recording ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  {recording ? (
                    <>
                      <span className="w-2 h-2 bg-red-600 rounded-full mr-2 animate-pulse"></span>
                      Recording: {formatTime(recordingTime)}
                    </>
                  ) : (
                    'Ready to record'
                  )}
                </span>
              </div>
            </div>
          </div>

          <div className="relative bg-black aspect-video">
            <Webcam
              audio={true}
              ref={webcamRef}
              className="w-full h-full object-contain"
              videoConstraints={{
                width: 1280,
                height: 720,
                facingMode: 'user',
              }}
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full pointer-events-none" // Ensure canvas doesn't block interactions
              style={{ zIndex: 10 }}
            />
            {recordingComplete && recordedChunks.length > 0 && (
              <div
                className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50"
                style={{ zIndex: 20 }}
              >
                <div className="text-white text-center">
                  <CheckCircle className="h-16 w-16 mx-auto text-green-400" />
                  <p className="mt-4 text-xl font-semibold">Recording Complete!</p>
                  <p className="mt-2">Duration: {formatTime(recordingTime)}</p>
                </div>
              </div>
            )}
          </div>

          <div className="p-6 bg-gray-50 border-t border-gray-200">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center space-x-4">
                {!recording && !recordingComplete && (
                  <button
                    onClick={handleStartRecording}
                    className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  >
                    <Play className="mr-2 h-5 w-5" />
                    Start Recording
                  </button>
                )}
                {recording && (
                  <button
                    onClick={handleStopRecording}
                    className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                  >
                    <StopCircle className="mr-2 h-5 w-5" />
                    Stop Recording
                  </button>
                )}
                {recordingComplete && !analysisComplete && (
                  <>
                    <button
                      onClick={handleAnalyze}
                      disabled={analyzing}
                      className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50"
                    >
                      {analyzing ? (
                        <>
                          <svg
                            className="animate-spin -ml-1 mr-2 h-5 w-5 text-white"
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                          >
                            <circle
                              className="opacity-25"
                              cx="12"
                              cy="12"
                              r="10"
                              stroke="currentColor"
                              strokeWidth="4"
                            ></circle>
                            <path
                              className="opacity-75"
                              fill="currentColor"
                              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                            ></path>
                          </svg>
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <BarChart2 className="mr-2 h-5 w-5" />
                          Analyze Presentation
                        </>
                      )}
                    </button>
                    <button
                      onClick={handleDownload}
                      className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                    >
                      <Save className="mr-2 h-5 w-5" />
                      Download Recording
                    </button>
                  </>
                )}
                {recordingComplete && (
                  <button
                    onClick={handleReset}
                    className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                  >
                    <Trash2 className="mr-2 h-5 w-5" />
                    Reset
                  </button>
                )}
              </div>
              {analysisComplete && (
                <Link
                  to="/dashboard"
                  className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                >
                  View Analysis Results
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Link>
              )}
            </div>
          </div>
        </div>

        {/* Real-time Feedback Dashboard */}
        {recording && (
          <div className="mt-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-medium text-gray-700">Posture Score</h4>
                <span className={`text-2xl font-bold ${postureScore >= 70 ? 'text-green-600' : postureScore >= 50 ? 'text-yellow-600' : 'text-red-600'}`}>
                  {postureScore}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full transition-all duration-300 ${postureScore >= 70 ? 'bg-green-600' : postureScore >= 50 ? 'bg-yellow-600' : 'bg-red-600'}`}
                  style={{ width: `${postureScore}%` }}
                ></div>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                {postureScore >= 70 ? 'Excellent posture!' : postureScore >= 50 ? 'Good, keep it up' : 'Stand straighter'}
              </p>
            </div>

            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-medium text-gray-700">Gestures</h4>
                <span className={`text-sm font-semibold ${gestureQuality.includes('Active') ? 'text-green-600' : 'text-gray-600'}`}>
                  {gestureQuality}
                </span>
              </div>
              <div className="flex items-center justify-center h-8">
                <div className={`w-3 h-3 rounded-full ${gestureQuality.includes('Both') ? 'bg-green-500 animate-pulse' : gestureQuality.includes('One') ? 'bg-yellow-500' : 'bg-gray-400'}`}></div>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                {gestureQuality.includes('Active') ? 'Great hand movements!' : 'Use more gestures'}
              </p>
            </div>

            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-medium text-gray-700">Speaking Pace</h4>
                <span className={`text-2xl font-bold ${speakingPace >= 120 && speakingPace <= 150 ? 'text-green-600' : 'text-yellow-600'}`}>
                  {speakingPace}
                </span>
              </div>
              <p className="text-xs text-gray-600">WPM (words/min)</p>
              <p className="text-xs text-gray-500 mt-2">
                {speakingPace === 0 ? 'Start speaking...' : speakingPace >= 120 && speakingPace <= 150 ? 'Perfect pace!' : speakingPace < 120 ? 'Speak a bit faster' : 'Slow down slightly'}
              </p>
            </div>

            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-medium text-gray-700">Volume Level</h4>
                <span className={`text-2xl font-bold ${volumeLevel >= 40 && volumeLevel <= 80 ? 'text-green-600' : 'text-yellow-600'}`}>
                  {volumeLevel}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full transition-all duration-150 ${volumeLevel >= 40 && volumeLevel <= 80 ? 'bg-green-600' : 'bg-yellow-600'}`}
                  style={{ width: `${volumeLevel}%` }}
                ></div>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                {volumeLevel < 40 ? 'Speak louder' : volumeLevel > 80 ? 'Lower your voice' : 'Good volume!'}
              </p>
            </div>
          </div>
        )}

        {/* Filler Words Counter */}
        {recording && fillerWordCount > 0 && (
          <div className="mt-4 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <div className="flex items-center">
              <MessageSquare className="h-5 w-5 text-yellow-600 mr-2" />
              <span className="text-sm font-medium text-yellow-800">
                Filler words detected: <span className="font-bold">{fillerWordCount}</span>
              </span>
              <span className="ml-2 text-xs text-yellow-600">
                (Try to pause instead of using "um", "uh", "like")
              </span>
            </div>
          </div>
        )}

        {/* Live Speech-to-Text Component */}
        {recording && (
          <div className="mt-6 bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-2">Live Transcript</h3>
            <div className="text-gray-700 text-base leading-relaxed max-h-40 overflow-y-auto">
              {transcript || 'Listening...'}
            </div>
          </div>
        )}

        {/* Recording Tips Section */}
        <div className="mt-8 bg-white rounded-lg shadow-lg overflow-hidden">
          <div className="p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Recording Tips</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="flex">
                <div className="flex-shrink-0">
                  <div className="flex items-center justify-center h-12 w-12 rounded-md bg-indigo-100 text-indigo-600">
                    <Video className="h-6 w-6" />
                  </div>
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-medium text-gray-900">Video Quality</h3>
                  <p className="mt-2 text-gray-600">
                    Ensure you're in a well-lit area with a neutral background. Position yourself so your upper body is visible.
                  </p>
                </div>
              </div>
              <div className="flex">
                <div className="flex-shrink-0">
                  <div className="flex items-center justify-center h-12 w-12 rounded-md bg-indigo-100 text-indigo-600">
                    <Mic className="h-6 w-6" />
                  </div>
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-medium text-gray-900">Audio Quality</h3>
                  <p className="mt-2 text-gray-600">
                    Record in a quiet environment. Speak clearly and at a moderate pace for the best analysis results.
                  </p>
                </div>
              </div>
              <div className="flex">
                <div className="flex-shrink-0">
                  <div className="flex items-center justify-center h-12 w-12 rounded-md bg-indigo-100 text-indigo-600">
                    <Info className="h-6 w-6" />
                  </div>
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-medium text-gray-900">Presentation Content</h3>
                  <p className="mt-2 text-gray-600">
                    Prepare your content in advance. Aim for a 3-5 minute presentation for the most effective feedback.
                  </p>
                </div>
              </div>
              <div className="flex">
                <div className="flex-shrink-0">
                  <div className="flex items-center justify-center h-12 w-12 rounded-md bg-indigo-100 text-indigo-600">
                    <CheckCircle className="h-6 w-6" />
                  </div>
                </div>
                <div className="ml-4">
                  <h3 className="text-lg font-medium text-gray-900">Best Practices</h3>
                  <p className="mt-2 text-gray-600">
                    Make eye contact with the camera. Use natural gestures and vary your tone to engage your audience.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Record;