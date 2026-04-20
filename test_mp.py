import sys
try:
    import mediapipe as mp
    print("mediapipe version:", getattr(mp, '__version__', 'unknown'))
    print("Has solutions?", hasattr(mp, 'solutions'))
    if not hasattr(mp, 'solutions'):
        import mediapipe.python.solutions.face_mesh as face_mesh
        print("Imported face_mesh directly!")
except Exception as e:
    print("Error:", e)
