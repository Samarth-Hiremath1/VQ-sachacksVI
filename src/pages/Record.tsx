import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Link } from 'react-router-dom';
import Webcam from 'react-webcam';
import { Mic, Video, StopCircle, Play, Save, Trash2, Info, CheckCircle, ArrowRight, BarChart2 } from 'lucide-react';
import { Pose } from '@mediapipe/pose';
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
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const poseRef = useRef<Pose | null>(null);
  const cameraRef = useRef<cameraUtils.Camera | null>(null);

  // Initialize MediaPipe Pose
  useEffect(() => {
    const pose = new Pose({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
      },
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

        // Clear previous drawings
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw pose landmarks and connections if detected
        if (results.poseLandmarks) {
          drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, {
            color: '#00FF00', // Green lines for connections
            lineWidth: 4,
          });
          drawLandmarks(ctx, results.poseLandmarks, {
            color: '#FF0000', // Red dots for landmarks
            lineWidth: 2,
            radius: 4,
          });
        } else {
          console.log('No pose landmarks detected');
        }
      }
    });

    poseRef.current = pose;

    return () => {
      pose.close();
    };
  }, []);

  // Start camera and pose detection when webcam is ready
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
        console.log('Camera and pose detection started');
      }
    };

    startCamera();

    return () => {
      if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
        console.log('Camera stopped');
      }
    };
  }, []);

  const handleStartRecording = useCallback(() => {
    setRecording(true);
    setRecordedChunks([]);
    setRecordingTime(0);
    setRecordingComplete(false);
    setAnalysisComplete(false);

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
      console.log('Recording started');

      timerRef.current = setInterval(() => {
        setRecordingTime((prevTime) => prevTime + 1);
      }, 1000);
    }
  }, []);

  const handleStopRecording = useCallback(() => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
    setRecording(false);
    setRecordingComplete(true);

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    console.log('Recording stopped');
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
            Capture your presentation with pose tracking for AI analysis and expert feedback.
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
              className="absolute top-0 left-0 w-full h-full"
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
                    Ensure you're in a well-lit area with a neutral background. Position yourself so your upper body
                    is visible.
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

// MediaPipe Pose connections
const POSE_CONNECTIONS: [number, number][] = [
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 12], [11, 23], [12, 24], [23, 24], [23, 25], [24, 26],
  [25, 27], [26, 28], [27, 29], [28, 30], [29, 31], [30, 32],
  [27, 31], [28, 32],
];

export default Record;