// Select all elements that should animate on scroll
const scrollElements = document.querySelectorAll('.scroll-animate');

const observerOptions = {
    root: null,
    rootMargin: '0px 0px -50px 0px',
    threshold: 0.1 // Trigger when 10% of the element is visible
};

const scrollObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('scrolled');
            entry.target.classList.remove('scrolled-up');
            entry.target.classList.remove('scrolled-down');
        } else {
            entry.target.classList.remove('scrolled');
            // Check if element left from the top or bottom
            if (entry.boundingClientRect.top < 0) {
                entry.target.classList.add('scrolled-up'); // left from top
            } else {
                entry.target.classList.add('scrolled-down'); // left from bottom
            }
        }
    });
}, observerOptions);

scrollElements.forEach(el => {
    scrollObserver.observe(el);
});
