// ===============================
// Contact Page JavaScript
// ===============================

document.addEventListener("DOMContentLoaded", function () {

    const form = document.querySelector("form");
    const nameInput = document.querySelector("input[name='name']");
    const emailInput = document.querySelector("input[name='email']");
    const messageInput = document.querySelector("textarea[name='message']");

    form.addEventListener("submit", function (event) {

        let valid = true;

        // Clear previous error messages
        clearErrors();

        // Name validation
        if (nameInput.value.trim() === "") {
            showError(nameInput, "Name is required");
            valid = false;
        }

        // Email validation
        if (!validateEmail(emailInput.value)) {
            showError(emailInput, "Enter a valid email address");
            valid = false;
        }

        // Message validation
        if (messageInput.value.trim().length < 10) {
            showError(messageInput, "Message must be at least 10 characters");
            valid = false;
        }

        if (!valid) {
            event.preventDefault();
        } else {
            showSuccess();
        }

    });

});

// ===============================
// Email Validation Function
// ===============================

function validateEmail(email) {
    const pattern = /^[^ ]+@[^ ]+\.[a-z]{2,3}$/;
    return pattern.test(email);
}

// ===============================
// Show Error Function
// ===============================

function showError(input, message) {
    const small = document.createElement("small");
    small.style.color = "red";
    small.innerText = message;
    input.parentElement.appendChild(small);
}

// ===============================
// Clear Previous Errors
// ===============================

function clearErrors() {
    const errors = document.querySelectorAll("small");
    errors.forEach(error => error.remove());
}

// ===============================
// Success Alert
// ===============================

function showSuccess() {
    alert("Message submitted successfully! Thank you for contacting us.");
}