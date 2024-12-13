from flask import render_template, Blueprint

admin = Blueprint('admin', __name__)

@admin.route('/admin/dashboard')
def dashboard():
    return render_template('admin/dashboard.html')

@admin.route('/admin/data-ingestion')
def data_ingestion():
    return render_template('admin/data_ingestion.html')

@admin.route('/admin/analytics')
def analytics():
    return render_template('admin/analytics.html')