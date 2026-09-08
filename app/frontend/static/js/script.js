// ==============================
// AI Career Platform - Frontend
// ==============================

// Resume Upload
function uploadResume() {

    const fileInput = document.querySelector("input[type='file']");

    if (fileInput.files.length === 0) {
        alert("Please select a resume.");
        return;
    }

    const file = fileInput.files[0];

    alert("Resume Uploaded Successfully!\n\nFile Name: " + file.name);
}

// Save User Details
function saveDetails() {

    let name = document.getElementById("name").value;
    let email = document.getElementById("email").value;
    let phone = document.getElementById("phone").value;
    let qualification = document.getElementById("qualification").value;
    let skills = document.getElementById("skills").value;
    let role = document.getElementById("role").value;

    if (
        name === "" ||
        email === "" ||
        phone === "" ||
        qualification === "" ||
        skills === "" ||
        role === ""
    ) {
        alert("Please fill all fields.");
        return;
    }

    alert("Details Saved Successfully!");
}

// Career Prediction (Dummy)
function predictCareer() {

    const careers = [
        "Data Scientist",
        "Software Developer",
        "Machine Learning Engineer",
        "Web Developer",
        "Cloud Engineer",
        "AI Engineer",
        "Cyber Security Analyst",
        "Database Administrator"
    ];

    let random = Math.floor(Math.random() * careers.length);

    let confidence = Math.floor(Math.random() * 11) + 90;

    document.getElementById("career").innerHTML = careers[random];

    document.getElementById("confidence").innerHTML = confidence + "%";

    document.getElementById("progressBar").style.width = confidence + "%";

    document.getElementById("progressBar").innerHTML = confidence + "%";
}

// Clear Form
function clearForm() {

    document.getElementById("name").value = "";
    document.getElementById("email").value = "";
    document.getElementById("phone").value = "";
    document.getElementById("qualification").value = "";
    document.getElementById("skills").value = "";
    document.getElementById("role").value = "";

}

// Welcome Message
window.onload = function () {

    console.log("AI Career Intelligence Platform Loaded");

};