document.addEventListener('keydown', function(event) {
    if (event.key.toLowerCase() === 'r') {
        const randomButton = document.querySelector('button[value="random"]');
        if (randomButton) randomButton.click();
    }
});