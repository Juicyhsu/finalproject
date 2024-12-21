from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as geni
import os
from azure.storage.blob import BlobServiceClient
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.core.exceptions import ResourceExistsError

# 從環境變數中讀取金鑰
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("環境變數 'GOOGLE_API_KEY' 未設定")

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
if not connection_string:
    raise ValueError("環境變數 'AZURE_STORAGE_CONNECTION_STRING' 未設定")

computer_vision_endpoint = os.getenv("AZURE_COMPUTER_VISION_ENDPOINT")
computer_vision_key = os.getenv("AZURE_COMPUTER_VISION_KEY")
if not computer_vision_endpoint or not computer_vision_key:
    raise ValueError("環境變數 'AZURE_COMPUTER_VISION_ENDPOINT' 或 'AZURE_COMPUTER_VISION_KEY' 未設定")

geni.configure(api_key=api_key)
model = geni.GenerativeModel("gemini-1.5-pro")

# 初始化 Azure Blob Storage
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_name = "chatbot-media"
container_client = blob_service_client.get_container_client(container_name)

# 初始化 Computer Vision 客戶端
computervision_client = ComputerVisionClient(
    computer_vision_endpoint,
    CognitiveServicesCredentials(computer_vision_key)
)

# 初始化 Flask 應用
app = Flask(__name__, static_url_path="", static_folder="static")


# 使用 SDK 呼叫圖片分析
def analyze_image(image_url):
    # 定義要提取的視覺特徵
    features = ["Description", "Tags"]
    # 呼叫 Computer Vision API
    analysis = computervision_client.analyze_image(image_url, features)
    
    # 解析結果
    description = analysis.description.captions[0].text if analysis.description.captions else "無法提供描述"
    # 確保標籤為字串列表
    tags = [tag.name for tag in analysis.tags] if analysis.tags else []
    return {
        "description": description,
        "tags": tags
    }

# 提供靜態 HTML 文件
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

# 路由：處理問題
@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        # 獲取請求的 JSON 數據
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' in request body"}), 400

        question = data["text"]

        # 調用生成模型
        response = model.generate_content(question)

        # 檢查模型返回的結果
        if not hasattr(response, 'text'):
            return jsonify({"error": "Invalid response from model"}), 500

        return jsonify({"response": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 路由：處理圖片上傳與分析
@app.route("/upload-image", methods=["POST"])
def upload_image():
    try:
        # 檢查是否有圖片上傳
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image file provided"}), 400
        
        # 上傳圖片到 Azure Blob Storage
        blob_client = container_client.get_blob_client(file.filename)
        try:
            blob_client.upload_blob(file, overwrite=True)
        except ResourceExistsError:
            return jsonify({"error": "Blob already exists and overwrite is not allowed"}), 400
        
        # 生成圖片的 URL
        image_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{file.filename}"


        # 使用 SDK 分析圖片
        analysis_result = analyze_image(image_url)
        # 檢查分析是否成功
        if "error" in analysis_result:
            return jsonify({"error": analysis_result["error"]}), 500

        description = analysis_result.get("description", "無描述")

        # 提取標籤的名稱
        tags = analysis_result.get("tags", [])


        # 構建 Gemini 分析提示
        prompt = f"這張圖片包含以下內容：描述為「{description}」，標籤包括：{', '.join(tags) if tags else '無標籤'}。請提供進一步的深入分析。"
        response = model.generate_content(prompt)
        
        # 確保模型有返回有效結果       
        if not hasattr(response, 'text'):
            return jsonify({"error": "Invalid response from model"}), 500  
       
        # 返回完整的分析結果
        return jsonify({
            "image_url": image_url,
            "description": description,
            "tags": tags,
            "gemini_analysis": response.text
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 啟動 Web 應用，debug 模式開啟
if __name__ == "__main__":        
    app.run(host="0.0.0.0", port=8000, debug=True)
