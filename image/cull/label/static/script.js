document.addEventListener('keydown', function(event) {
    if (event.key.toLowerCase() === 'r') {
        const randomButton = document.querySelector('button[value="random"]');
        if (randomButton) randomButton.click();
    }
    if (event.key === 'ArrowLeft') {
        const prevButton = document.querySelector('button[value="prev"]');
        if (prevButton) prevButton.click();
    }
    if (event.key === 'ArrowRight') {
        const nextButton = document.querySelector('button[value="next"]');
        if (nextButton) nextButton.click();
    }
});

const randomOnLabelToggle = document.getElementById('randomOnLabel');
if (randomOnLabelToggle) {
    randomOnLabelToggle.addEventListener('change', () => {
        const form = document.getElementById('cullForm');
        if (form) form.submit();
    });
}

const labelledNavRadios = document.querySelectorAll('input[name="labelled_nav_mode"]');
if (labelledNavRadios.length) {
    labelledNavRadios.forEach((radio) => {
        radio.addEventListener('change', () => {
            const form = document.getElementById('cullForm');
            if (form) form.submit();
        });
    });
}
