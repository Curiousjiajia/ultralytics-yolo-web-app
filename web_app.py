import base64
import os
import uuid
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_from_directory

from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

print("Loading YOLO model...")
model = YOLO("yolo26n.pt")
print("✅ Model loaded!\n")

MAX_UPLOAD_IMAGES = 4


@app.route("/")
def index():
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO 检测</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            min-height: 100vh; 
            padding: 20px; 
        }
        .container { 
            max-width: 1400px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 20px; 
            padding: 40px; 
            box-shadow: 0 20px 60px rgba(0,0,0,0.3); 
        }
        h1 { 
            text-align: center; 
            color: #333; 
            margin-bottom: 10px; 
            font-size: clamp(1.5rem, 4vw, 2.5rem);
        }
        .subtitle { 
            text-align: center; 
            color: #666; 
            margin-bottom: 30px; 
            font-size: clamp(0.9rem, 2vw, 1.1rem);
        }
        #uploadBox { 
            border: 3px dashed #667eea; 
            border-radius: 15px; 
            padding: clamp(30px, 5vw, 50px); 
            text-align: center; 
            cursor: pointer; 
            background: #f8f9ff; 
            transition: all 0.3s; 
        }
        #uploadBox:hover { 
            background: #eef0ff; 
            border-color: #764ba2; 
            transform: scale(1.02);
        }
        #fileInput { display: none; }
        .btn { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            border: none; 
            padding: 12px 30px; 
            border-radius: 25px; 
            cursor: pointer; 
            font-size: 16px; 
            margin: 10px 5px; 
            transition: all 0.3s;
        }
        .btn:hover { 
            opacity: 0.9; 
            transform: translateY(-2px); 
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .btn-secondary { background: #6c757d; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        #previewArea, #resultArea { margin-top: 30px; display: none; }
        
        /* 响应式网格布局 */
        .image-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); 
            gap: clamp(15px, 3vw, 25px); 
            margin-top: 20px; 
        }
        
        .image-item { 
            border: 1px solid #ddd; 
            border-radius: 10px; 
            padding: 15px; 
            background: #fff; 
            position: relative;
            transition: all 0.3s;
        }
        .image-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }
        .image-item img { 
            width: 100%; 
            height: auto;
            border-radius: 8px; 
            display: block;
        }
        .image-count { 
            position: absolute; 
            top: -10px; 
            right: -10px; 
            background: #667eea; 
            color: white; 
            width: 30px; 
            height: 30px; 
            border-radius: 50%; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            font-weight: bold; 
            font-size: 14px;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
        }
        #backBtn { 
            position: fixed; 
            top: 20px; 
            left: 20px; 
            background: white; 
            border: 2px solid #667eea; 
            color: #667eea; 
            padding: 10px 20px; 
            border-radius: 25px; 
            cursor: pointer; 
            font-weight: bold; 
            display: none; 
            z-index: 1000;
            transition: all 0.3s;
        }
        #backBtn:hover { 
            background: #667eea; 
            color: white; 
            transform: translateY(-2px);
        }
        #loading { 
            display: none; 
            text-align: center; 
            padding: 40px; 
        }
        .spinner { 
            border: 4px solid #f3f3f3; 
            border-top: 4px solid #667eea; 
            border-radius: 50%; 
            width: 50px; 
            height: 50px; 
            animation: spin 1s linear infinite; 
            margin: 0 auto 20px; 
        }
        @keyframes spin { 
            0% { transform: rotate(0deg); } 
            100% { transform: rotate(360deg); } 
        }
        .detection-info { 
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 8px; 
            margin-top: 10px; 
        }
        .badge { 
            display: inline-block; 
            padding: 4px 8px; 
            border-radius: 12px; 
            font-size: 12px; 
            margin: 2px; 
        }
        .badge-primary { background: #667eea; color: white; }
        .badge-success { background: #28a745; color: white; }
        .badge-info { background: #17a2b8; color: white; }
        .limit-info { 
            background: #fff3cd; 
            border: 1px solid #ffc107; 
            color: #856404; 
            padding: 10px 15px; 
            border-radius: 8px; 
            margin-top: 15px; 
            text-align: center; 
        }
        
        /* 响应式调整 */
        @media (max-width: 768px) {
            .container {
                padding: 20px;
                margin-top: 10px;
            }
            .image-grid {
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
            }
            #backBtn {
                top: 10px;
                left: 10px;
                padding: 8px 15px;
                font-size: 14px;
            }
            .btn {
                padding: 10px 20px;
                font-size: 14px;
                margin: 5px;
            }
        }
        
        @media (max-width: 480px) {
            body {
                padding: 10px;
            }
            .container {
                padding: 15px;
                border-radius: 15px;
            }
            .image-grid {
                grid-template-columns: 1fr;
                gap: 12px;
            }
            h1 {
                font-size: 1.5rem;
            }
            #uploadBox {
                padding: 30px 20px;
            }
        }
        
        @media (min-width: 1200px) {
            .image-grid {
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            }
        }
        
        @media (min-width: 1600px) {
            .container {
                max-width: 1600px;
            }
            .image-grid {
                grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            }
        }
    </style>
</head>
<body>
    <button id="backBtn" onclick="resetAll()">← 返回</button>
    <div class="container">
        <h1>🎯 YOLO 目标检测</h1>
        <p class="subtitle">支持批量检测 | 每次最多4张图片 | 自适应布局</p>
        
        <div id="uploadSection">
            <div id="uploadBox">
                <h2>📷 点击选择图片</h2>
                <p style="color: #666; margin-top: 10px;">或拖拽图片到此处</p>
                <p style="color: #667eea; margin-top: 5px; font-weight: bold;">最多选择 4 张图片</p>
                <input type="file" id="fileInput" accept="image/*" multiple>
            </div>
            
            <div id="limitInfo" class="limit-info" style="display: none;">
                ⚠️ 已选择 <span id="selectedCount">0</span>/4 张图片
            </div>
            
            <div id="previewArea">
                <h3 style="margin-top: 30px;">已选择的图片：</h3>
                <div id="previewGrid" class="image-grid"></div>
                <div style="text-align: center; margin-top: 20px;">
                    <button class="btn" id="detectBtn" onclick="startDetection()">🔍 开始检测</button>
                    <button class="btn btn-secondary" onclick="resetAll()">🗑️ 清空</button>
                </div>
            </div>
        </div>
        
        <div id="loading">
            <div class="spinner"></div>
            <p>正在检测中...</p>
            <p id="progressText" style="color: #666; margin-top: 10px;"></p>
        </div>
        
        <div id="resultArea">
            <h3 style="margin-bottom: 20px;">✅ 检测结果</h3>
            <div id="resultGrid" class="image-grid"></div>
        </div>
    </div>

    <script>
        var selectedFiles = [];
        var results = [];
        var MAX_IMAGES = 4;

        document.getElementById('uploadBox').addEventListener('click', function() {
            document.getElementById('fileInput').click();
        });

        document.getElementById('fileInput').addEventListener('change', function(e) {
            handleFiles(e.target.files);
        });

        function handleFiles(files) {
            console.log('Files selected:', files.length);
            
            var imageFiles = Array.from(files).filter(function(f) { 
                return f.type.startsWith('image/'); 
            });
            
            if (imageFiles.length === 0) {
                alert('请选择图片文件！');
                return;
            }
            
            if (imageFiles.length > MAX_IMAGES) {
                alert('最多只能选择 ' + MAX_IMAGES + ' 张图片！\\n当前选择了 ' + imageFiles.length + ' 张。\\n将只保留前 ' + MAX_IMAGES + ' 张。');
                imageFiles = imageFiles.slice(0, MAX_IMAGES);
            }
            
            selectedFiles = imageFiles;
            
            updateLimitInfo();
            showPreviews();
        }

        function updateLimitInfo() {
            var limitInfo = document.getElementById('limitInfo');
            var selectedCount = document.getElementById('selectedCount');
            
            if (selectedFiles.length > 0) {
                limitInfo.style.display = 'block';
                selectedCount.textContent = selectedFiles.length;
                
                if (selectedFiles.length >= MAX_IMAGES) {
                    limitInfo.style.background = '#f8d7da';
                    limitInfo.style.borderColor = '#f5c6cb';
                    limitInfo.style.color = '#721c24';
                } else {
                    limitInfo.style.background = '#fff3cd';
                    limitInfo.style.borderColor = '#ffc107';
                    limitInfo.style.color = '#856404';
                }
            } else {
                limitInfo.style.display = 'none';
            }
        }

        function showPreviews() {
            var previewGrid = document.getElementById('previewGrid');
            previewGrid.innerHTML = '';
            
            selectedFiles.forEach(function(file, idx) {
                var reader = new FileReader();
                reader.onload = function(e) {
                    var div = document.createElement('div');
                    div.className = 'image-item';
                    div.innerHTML = 
                        '<div class="image-count">' + (idx + 1) + '</div>' +
                        '<img src="' + e.target.result + '">' +
                        '<p style="margin-top: 5px; font-size: 12px; color: #666; word-break: break-all;">' + file.name + '</p>';
                    previewGrid.appendChild(div);
                };
                reader.readAsDataURL(file);
            });
            
            document.getElementById('uploadBox').style.display = 'none';
            document.getElementById('previewArea').style.display = 'block';
        }

        async function startDetection() {
            if (selectedFiles.length === 0) {
                alert('请先选择图片！');
                return;
            }

            document.getElementById('uploadSection').style.display = 'none';
            document.getElementById('loading').style.display = 'block';
            document.getElementById('backBtn').style.display = 'block';
            
            results = [];
            var total = selectedFiles.length;
            
            for (var i = 0; i < total; i++) {
                document.getElementById('progressText').textContent = '处理进度: ' + (i + 1) + '/' + total;
                
                var formData = new FormData();
                formData.append('image', selectedFiles[i]);
                
                try {
                    var response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    var data = await response.json();
                    if (data.success) {
                        results.push({
                            filename: selectedFiles[i].name,
                            ...data
                        });
                    }
                } catch (err) {
                    console.error('Error:', err);
                }
            }
            
            showResults();
        }

        function showResults() {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('resultArea').style.display = 'block';
            
            var resultGrid = document.getElementById('resultGrid');
            resultGrid.innerHTML = '';
            
            results.forEach(function(result, idx) {
                var div = document.createElement('div');
                div.className = 'image-item';
                
                var detectionsHtml = '';
                if (result.detections && result.detections.length > 0) {
                    detectionsHtml = result.detections.map(function(d, i) {
                        return '<div style="background: white; padding: 5px; margin: 5px 0; border-radius: 4px; font-size: 13px;">' +
                            '<strong>' + (i+1) + '. ' + d.class + '</strong> ' +
                            '<span class="badge badge-success">' + (d.confidence * 100).toFixed(1) + '%</span>' +
                            '</div>';
                    }).join('');
                } else {
                    detectionsHtml = '<p style="color: #999; font-size: 13px;">未检测到对象</p>';
                }
                
                var classesBadge = result.classes_detected.map(function(c) {
                    return '<span class="badge badge-info">' + c + '</span>';
                }).join(' ');
                
                div.innerHTML = 
                    '<div class="image-count">' + (idx+1) + '</div>' +
                    '<h4 style="margin-bottom: 10px; font-size: 14px; word-break: break-all;">' + result.filename + '</h4>' +
                    '<img src="' + result.result_image + '">' +
                    '<div class="detection-info">' +
                    '<p><strong>检测到:</strong> <span class="badge badge-primary">' + result.num_detections + ' 个对象</span></p>' +
                    '<p style="margin: 10px 0;"><strong>类别:</strong><br>' + classesBadge + '</p>' +
                    detectionsHtml +
                    '<button class="btn" style="width: 100%; margin-top: 10px; padding: 8px;" onclick="downloadResult(\\'' + result.result_id + '\\')">💾 下载结果</button>' +
                    '</div>';
                
                resultGrid.appendChild(div);
            });
        }

        function downloadResult(resultId) {
            window.open('/results/' + resultId + '_result.jpg', '_blank');
        }

        function resetAll() {
            document.getElementById('resultArea').style.display = 'none';
            document.getElementById('uploadSection').style.display = 'block';
            document.getElementById('loading').style.display = 'none';
            document.getElementById('backBtn').style.display = 'none';
            document.getElementById('previewArea').style.display = 'none';
            document.getElementById('uploadBox').style.display = 'block';
            document.getElementById('limitInfo').style.display = 'none';
            document.getElementById('fileInput').value = '';
            selectedFiles = [];
            results = [];
        }

        // Drag and drop support
        var uploadBox = document.getElementById('uploadBox');
        
        uploadBox.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadBox.style.background = '#e0e4ff';
        });
        
        uploadBox.addEventListener('dragleave', function() {
            uploadBox.style.background = '#f8f9ff';
        });
        
        uploadBox.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadBox.style.background = '#f8f9ff';
            if (e.dataTransfer.files.length > 0) {
                handleFiles(e.dataTransfer.files);
            }
        });
    </script>
</body>
</html>"""
    return html_content


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower() if "." in file.filename else "jpg"
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    upload_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(upload_path)

    try:
        results = model(upload_path, verbose=False)
        result = results[0]

        output_name = f"{Path(upload_path).stem}_result.jpg"
        output_path = os.path.join(RESULTS_FOLDER, output_name)
        annotated = result.plot()
        cv2.imwrite(output_path, annotated)

        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, c, cls_id in zip(boxes, confs, cls_ids):
                detections.append(
                    {
                        "class": result.names[cls_id],
                        "confidence": float(c),
                        "bbox": {"x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3])},
                    }
                )

        with open(output_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        result_id = str(uuid.uuid4())

        return jsonify(
            {
                "success": True,
                "result_id": result_id,
                "result_image": f"data:image/jpeg;base64,{b64}",
                "detections": detections,
                "num_detections": len(detections),
                "classes_detected": list(set([d["class"] for d in detections])),
            }
        )
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/results/<filename>")
def serve_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 YOLO Web Application")
    print("=" * 60)
    print(f"📸 Max images per upload: {MAX_UPLOAD_IMAGES}")
    print("🌐 Open: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
