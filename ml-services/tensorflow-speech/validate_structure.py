#!/usr/bin/env python3
"""
Validation script for TensorFlow Speech Analysis Service structure.
This script validates the code structure without requiring heavy dependencies.
"""

import os
import sys
import ast
import importlib.util

def validate_file_structure():
    """Validate that all required files exist."""
    required_files = [
        'app/__init__.py',
        'app/main.py',
        'app/models/__init__.py',
        'app/models/speech_quality_analyzer.py',
        'app/models/filler_word_detector.py',
        'app/services/__init__.py',
        'app/services/audio_processor.py',
        'app/services/web_speech_integration.py',
        'app/services/speech_analysis_service.py',
        'requirements.txt',
        'Dockerfile'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files exist")
        return True

def validate_python_syntax():
    """Validate Python syntax for all Python files."""
    python_files = []
    for root, dirs, files in os.walk('app'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            ast.parse(content)
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
    
    if syntax_errors:
        print(f"✗ Syntax errors found:")
        for error in syntax_errors:
            print(f"  {error}")
        return False
    else:
        print("✓ All Python files have valid syntax")
        return True

def validate_imports():
    """Validate that imports are structured correctly."""
    try:
        # Check main.py imports
        with open('app/main.py', 'r') as f:
            main_content = f.read()
        
        required_imports = [
            'from fastapi import FastAPI',
            'from .models.speech_quality_analyzer import',
            'from .models.filler_word_detector import',
            'from .services.audio_processor import',
            'from .services.web_speech_integration import',
            'from .services.speech_analysis_service import'
        ]
        
        missing_imports = []
        for import_stmt in required_imports:
            if import_stmt not in main_content:
                missing_imports.append(import_stmt)
        
        if missing_imports:
            print(f"✗ Missing imports in main.py: {missing_imports}")
            return False
        else:
            print("✓ All required imports found in main.py")
            return True
            
    except Exception as e:
        print(f"✗ Error validating imports: {e}")
        return False

def validate_class_definitions():
    """Validate that key classes are defined."""
    class_files = {
        'app/models/speech_quality_analyzer.py': ['SpeechQualityAnalyzer', 'SpeechQualityMetrics'],
        'app/models/filler_word_detector.py': ['FillerWordDetector', 'FillerWordAnalysis'],
        'app/services/audio_processor.py': ['AudioProcessor'],
        'app/services/web_speech_integration.py': ['WebSpeechAPIIntegration'],
        'app/services/speech_analysis_service.py': ['SpeechAnalysisService', 'ComprehensiveSpeechAnalysis']
    }
    
    missing_classes = []
    for file_path, expected_classes in class_files.items():
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            for class_name in expected_classes:
                if f"class {class_name}" not in content:
                    missing_classes.append(f"{file_path}: {class_name}")
        except Exception as e:
            missing_classes.append(f"{file_path}: Error reading file - {e}")
    
    if missing_classes:
        print(f"✗ Missing class definitions:")
        for missing in missing_classes:
            print(f"  {missing}")
        return False
    else:
        print("✓ All required classes found")
        return True

def validate_fastapi_endpoints():
    """Validate that FastAPI endpoints are defined."""
    try:
        with open('app/main.py', 'r') as f:
            main_content = f.read()
        
        required_endpoints = [
            ('@app.get("/"', 'root endpoint'),
            ('@app.get("/health"', 'health check endpoint'),
            ('@app.post("/analyze-audio-file"', 'audio file analysis endpoint'),
            ('@app.post("/analyze-speech"', 'speech analysis endpoint'),
            ('@app.post("/analyze-comprehensive"', 'comprehensive analysis endpoint'),
            ('@app.post("/real-time-feedback"', 'real-time feedback endpoint'),
            ('@app.get("/service/status"', 'service status endpoint'),
            ('@app.get("/models/info"', 'model info endpoint')
        ]
        
        missing_endpoints = []
        for endpoint_pattern, description in required_endpoints:
            if endpoint_pattern not in main_content:
                missing_endpoints.append(f"{description} ({endpoint_pattern})")
        
        if missing_endpoints:
            print(f"✗ Missing FastAPI endpoints:")
            for missing in missing_endpoints:
                print(f"  {missing}")
            return False
        else:
            print("✓ All required FastAPI endpoints found")
            return True
            
    except Exception as e:
        print(f"✗ Error validating endpoints: {e}")
        return False

def main():
    """Run all validation checks."""
    print("Validating TensorFlow Speech Analysis Service...")
    print("=" * 50)
    
    # Change to the service directory
    service_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(service_dir)
    
    checks = [
        validate_file_structure,
        validate_python_syntax,
        validate_imports,
        validate_class_definitions,
        validate_fastapi_endpoints
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 All validation checks passed!")
        print("The TensorFlow Speech Analysis Service is properly structured and ready for deployment.")
        return 0
    else:
        print("❌ Some validation checks failed.")
        print("Please fix the issues above before deploying the service.")
        return 1

if __name__ == "__main__":
    sys.exit(main())