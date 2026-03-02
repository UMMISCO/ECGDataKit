(function () {
  'use strict';

  // ── Hamburger toggle with overlay ──
  var hamburger = document.querySelector('.hamburger');
  var sidebar = document.querySelector('.sidebar');
  var overlay = document.querySelector('.sidebar-overlay');

  if (hamburger && sidebar) {
    hamburger.addEventListener('click', function () {
      sidebar.classList.toggle('open');
      if (overlay) overlay.classList.toggle('open');
    });
  }

  if (overlay) {
    overlay.addEventListener('click', function () {
      if (sidebar) sidebar.classList.remove('open');
      overlay.classList.remove('open');
    });
  }

  // ── Wrap tables in responsive scroll container ──
  document.querySelectorAll('.main-content table').forEach(function (table) {
    if (table.parentNode.classList.contains('table-responsive')) return;
    var wrapper = document.createElement('div');
    wrapper.className = 'table-responsive';
    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(table);
  });

  // ── Copy buttons on code blocks ──
  document.querySelectorAll('pre').forEach(function (pre) {
    // Skip if already has a copy button (e.g. hero install)
    if (pre.querySelector('.copy-btn')) return;

    var btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.setAttribute('aria-label', 'Copy code');

    btn.addEventListener('click', function () {
      var code = pre.querySelector('code');
      var text = code ? code.textContent : pre.textContent;
      navigator.clipboard.writeText(text).then(function () {
        btn.textContent = 'Copied!';
        btn.classList.add('copied');
        setTimeout(function () {
          btn.textContent = 'Copy';
          btn.classList.remove('copied');
        }, 2000);
      });
    });

    // Place button inside the highlight wrapper if it exists, else in pre
    var highlight = pre.closest('.highlight');
    if (highlight) {
      highlight.style.position = 'relative';
      highlight.appendChild(btn);
    } else {
      pre.appendChild(btn);
    }
  });

  // ── Smooth scroll offset for anchor links ──
  document.querySelectorAll('a[href^="#"]').forEach(function (a) {
    a.addEventListener('click', function (e) {
      var target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        var top = target.getBoundingClientRect().top + window.scrollY - 80;
        window.scrollTo({ top: top, behavior: 'smooth' });
      }
    });
  });

  // ── TOC Scroll Spy ──
  var tocNav = document.querySelector('.toc-sidebar nav');
  if (tocNav) {
    var tocLinks = Array.prototype.slice.call(tocNav.querySelectorAll('a[href^="#"]'));
    var headingIds = [];
    tocLinks.forEach(function (link) {
      var id = link.getAttribute('href').slice(1);
      if (id) headingIds.push(id);
    });

    var headings = headingIds.map(function (id) {
      return document.getElementById(id);
    }).filter(Boolean);

    if (headings.length) {
      var scrollTimeout;
      var offset = 100; // header height + some margin

      function updateActiveHeading() {
        var scrollY = window.scrollY;
        var active = null;

        for (var i = 0; i < headings.length; i++) {
          if (headings[i].getBoundingClientRect().top + window.scrollY - offset <= scrollY) {
            active = headings[i];
          } else {
            break;
          }
        }

        // If scrolled to bottom, activate last heading
        if (window.innerHeight + scrollY >= document.body.scrollHeight - 10) {
          active = headings[headings.length - 1];
        }

        tocLinks.forEach(function (link) {
          link.classList.remove('active');
        });

        if (active) {
          var activeLink = tocNav.querySelector('a[href="#' + active.id + '"]');
          if (activeLink) activeLink.classList.add('active');
        }
      }

      window.addEventListener('scroll', function () {
        if (scrollTimeout) cancelAnimationFrame(scrollTimeout);
        scrollTimeout = requestAnimationFrame(updateActiveHeading);
      }, { passive: true });

      // Run once on load
      updateActiveHeading();
    }
  }
})();
