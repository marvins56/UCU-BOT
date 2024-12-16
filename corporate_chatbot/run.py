# from app import create_app
# import os

# # Set environment variables
# os.environ['USER_AGENT'] = 'CorporateChatbot/1.0'
# os.environ['FLASK_DEBUG'] = '0'
# os.environ['FLASK_ENV'] = 'development'

# app = create_app()

# if __name__ == '__main__':
#     try:
#         print("Starting application...")
#         app.run(
#             host='127.0.0.1', 
#             port=8000, 
#             debug=True,
#             use_reloader=False  # Disable auto-reloader
#         )
#     except Exception as e:
#         print(f"Failed to start application: {str(e)}")

from app import create_app
import os
from waitress import serve

# Set environment variables
os.environ['USER_AGENT'] = 'CorporateChatbot/1.0'

# Create the application instance
app = create_app()

if __name__ == '__main__':
    if app is None:
        print("Application failed to start due to missing models.")
    else:
        try:
            PORT = 5501
            print(f"Starting application on port {PORT}...")
            serve(app, host='127.0.0.1', port=PORT)
        except Exception as e:
            print(f"Failed to start application: {str(e)}")
        finally:
            if hasattr(app, 'controller'):
                app.controller.cleanup()