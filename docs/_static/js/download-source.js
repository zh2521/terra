// Make the theme's "Download source file" button actually download the file
// (e.g. the .ipynb notebook) instead of opening it inline in a new browser tab.
// The HTML5 `download` attribute forces a same-origin link to download.
document.addEventListener("DOMContentLoaded", function () {
  document
    .querySelectorAll("a.btn-download-source-button")
    .forEach(function (link) {
      link.setAttribute("download", "");
      link.removeAttribute("target");
    });
});
