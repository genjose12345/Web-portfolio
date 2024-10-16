export function initSkillsAnimation() {
    const skillsSection = document.getElementById('skills');
    const skillBars = document.querySelectorAll('.skill');
    let animated = false;

    function animateSkills() {
        const sectionTop = skillsSection.getBoundingClientRect().top;
        const windowHeight = window.innerHeight;

        if (sectionTop < windowHeight * 0.75 && !animated) {
            skillBars.forEach(bar => {
                bar.classList.add('animate');
            });
            animated = true;
        }
    }

    window.addEventListener('scroll', animateSkills);
    animateSkills(); // Check on load in case the skills section is already in view
}