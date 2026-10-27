// main.js
// ==============================================================
// Global JavaScript for CrimePortal
// Loaded on every page via base.html
// ==============================================================

// ----------------------------------------------------------
// 1. Auto-dismiss flash alerts after 5 seconds
// ----------------------------------------------------------
document.addEventListener('DOMContentLoaded', function () {

    const alerts = document.querySelectorAll('.alert.alert-dismissible');
    alerts.forEach(function (alert) {
        setTimeout(function () {
            // Bootstrap 5 dismiss
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            if (bsAlert) bsAlert.close();
        }, 5000);  // 5000ms = 5 seconds
    });

    // ----------------------------------------------------------
    // 2. Highlight active nav link based on current URL
    // ----------------------------------------------------------
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(function (link) {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });

    // ----------------------------------------------------------
    // 3. File upload preview — show selected filename
    // ----------------------------------------------------------
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
        fileInput.addEventListener('change', function () {
            const fileName = this.files[0]
                ? this.files[0].name
                : 'No file chosen';
            // Show file name next to the input
            let preview = document.getElementById('file-preview');
            if (!preview) {
                preview = document.createElement('small');
                preview.id = 'file-preview';
                preview.className = 'text-success d-block mt-1';
                this.parentNode.appendChild(preview);
            }
            preview.textContent = '✓ Selected: ' + fileName;
        });
    }

    // ----------------------------------------------------------
    // 4. Confirm before form submission on critical forms
    //    (any form with data-confirm attribute)
    // ----------------------------------------------------------
    document.querySelectorAll('form[data-confirm]').forEach(function (form) {
        form.addEventListener('submit', function (e) {
            const msg = form.dataset.confirm ||
                        'Are you sure you want to submit?';
            if (!confirm(msg)) {
                e.preventDefault();
            }
        });
    });

    // ----------------------------------------------------------
    // 5. Smooth scroll-to-top button
    //    (appears when user scrolls down 300px)
    // ----------------------------------------------------------
    const scrollBtn = document.getElementById('scrollTopBtn');
    if (scrollBtn) {
        window.addEventListener('scroll', function () {
            scrollBtn.style.display =
                window.scrollY > 300 ? 'block' : 'none';
        });
        scrollBtn.addEventListener('click', function () {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

});  // end DOMContentLoaded