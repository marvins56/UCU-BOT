from app import create_app

app = create_app()

if __name__ == '__main__':
    if app is None:
        print("Application failed to start due to missing models.")
    else:
        app.run(debug=True)