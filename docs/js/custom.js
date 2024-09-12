document.addEventListener("DOMContentLoaded", function() {
    // Select all div elements with the class 'md-copyright'
    var copyrightDivs = document.querySelectorAll('.md-copyright');

    // Loop through each selected div and remove its text content
    copyrightDivs.forEach(function(div) {
        div.innerHTML = ''; // Clear the content of each div
    });
});