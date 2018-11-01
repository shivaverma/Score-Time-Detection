import os
from final_project import create_result
from flask import Flask, render_template, request, jsonify, url_for

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('data')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], "demo.png")
    
    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    file.save(f)
    data = {
        'status': 'success',
        'result': create_result()
    }
    return render_template('index.html', data=data)


if __name__ == "__main__":
    
    app.run(port=8080)
