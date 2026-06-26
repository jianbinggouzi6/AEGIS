async function uploadFile() {

    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("请选择音频文件");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    document.getElementById("filename").innerText = "File: " + data.filename;
    document.getElementById("score").innerText = "Score: " + data.score;
    document.getElementById("level").innerText = "Level: " + data.level;
    document.getElementById("advice").innerText = "Advice: " + data.advice;

    const lamp = document.getElementById("lamp");

    if (data.level === "normal") {
        lamp.style.background = "green";
    } else if (data.level === "warning") {
        lamp.style.background = "orange";
    } else {
        lamp.style.background = "red";
    }
}