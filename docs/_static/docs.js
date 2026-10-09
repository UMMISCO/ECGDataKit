// Doc pages: collapsible sidebar groups with icons, Ctrl/Cmd+K focuses the search box
(function () {
  var ICONS = {
    "Guides": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>',
    "Docs": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h8"/>',
    "Project": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
  };
  var CHEVRON = '<svg class="nav-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M6 9l6 6 6-6"/></svg>';
  var STORE = "ecgdatakit-collapsed";

  function saved() {
    try { return JSON.parse(localStorage.getItem(STORE) || "[]"); } catch (e) { return []; }
  }

  document.addEventListener("DOMContentLoaded", function () {
    var collapsed = saved();
    document.querySelectorAll(".sidebar-tree p.caption").forEach(function (caption) {
      var text = caption.textContent.trim();
      var icon = ICONS[text] || ICONS["Docs"];
      caption.insertAdjacentHTML("afterbegin",
        '<svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' + icon + "</svg>");
      caption.insertAdjacentHTML("beforeend", CHEVRON);
      var list = caption.nextElementSibling;
      var holdsCurrent = list && list.querySelector(".current-page");
      if (collapsed.indexOf(text) >= 0 && !holdsCurrent) caption.classList.add("collapsed");
      caption.addEventListener("click", function () {
        caption.classList.toggle("collapsed");
        var now = saved().filter(function (t) { return t !== text; });
        if (caption.classList.contains("collapsed")) now.push(text);
        try { localStorage.setItem(STORE, JSON.stringify(now)); } catch (e) {}
      });
    });

    var input = document.querySelector(".docs-search input");
    if (input) {
      document.addEventListener("keydown", function (event) {
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
          event.preventDefault();
          input.focus();
        }
      });
      if (/Mac|iPhone|iPad/.test(navigator.platform)) {
        var kbd = document.querySelector(".docs-search kbd");
        if (kbd) kbd.textContent = "⌘ K";
      }
    }
  });
})();
