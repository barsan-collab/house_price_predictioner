/**
 * script.js
 * ---------
 * Interactive AJAX-powered frontend for the House Price Predictor.
 * Handles form submission via fetch(), toast notifications, input validation,
 * range-slider sync, and dynamic result rendering — all without page reloads.
 */

(function () {
    'use strict';

    // ---- DOM references ----
    const predictionForm = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn?.querySelector('.btn-text');
    const btnSpinner = submitBtn?.querySelector('.btn-spinner');
    const resultContainer = document.getElementById('result-container');
    const toastContainer = document.getElementById('toast-container');
    const comparisonContainer = document.getElementById('comparison-container');
    
    // Comparison State
    let compareList = [];
    try {
        const stored = localStorage.getItem('compareList');
        if (stored) compareList = JSON.parse(stored);
    } catch(e) {}
    let currentPrediction = null;

    // Range sliders
    const bedroomRange = document.getElementById('bedrooms');
    const bedroomValue = document.getElementById('bedrooms-value');
    const bathroomRange = document.getElementById('bathrooms');
    const bathroomValue = document.getElementById('bathrooms-value');

    // ---- Toast notification system ----
    function showToast(message, type = 'info', duration = 4000) {
        if (!toastContainer) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;

        const icons = { success: '✓', error: '✕', info: 'ℹ', warning: '⚠' };
        toast.innerHTML = `<span>${icons[type] || 'ℹ'}</span><span>${message}</span>`;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('toast-exit');
            toast.addEventListener('animationend', () => toast.remove());
        }, duration);
    }

    // ---- Range slider sync ----
    function syncRangeDisplay(range, display) {
        if (!range || !display) return;
        display.textContent = range.value;
        range.addEventListener('input', () => {
            display.textContent = range.value;
        });
    }

    syncRangeDisplay(bedroomRange, bedroomValue);
    syncRangeDisplay(bathroomRange, bathroomValue);

    // ---- Area input formatting ----
    const areaInput = document.getElementById('area');
    if (areaInput) {
        areaInput.addEventListener('input', () => {
            const val = parseFloat(areaInput.value);
            const hint = document.getElementById('area-hint');
            if (hint && !isNaN(val) && val > 0) {
                const sqm = (val * 0.0929).toFixed(1);
                hint.textContent = `≈ ${sqm} sq. meters`;
            } else if (hint) {
                hint.textContent = '';
            }
        });
    }

    // ---- Form validation highlight ----
    function validateForm() {
        const area = parseFloat(document.getElementById('area')?.value);
        const location = document.getElementById('location')?.value;

        if (!area || area <= 0) {
            showToast('Please enter a valid area.', 'error');
            document.getElementById('area')?.focus();
            return false;
        }

        if (area > 1000000) {
            showToast('Area seems unrealistically large. Please check.', 'error');
            document.getElementById('area')?.focus();
            return false;
        }

        if (!location) {
            showToast('Please select a location.', 'error');
            document.getElementById('location')?.focus();
            return false;
        }

        return true;
    }

    // ---- Animate number counting ----
    function animateCounter(element, targetText) {
        // Extract numeric value from formatted price string
        const numMatch = targetText.replace(/[^\d.]/g, '');
        const targetNum = parseFloat(numMatch);

        if (isNaN(targetNum) || targetNum === 0) {
            element.textContent = targetText;
            return;
        }

        const duration = 1200;
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease-out cubic
            const eased = 1 - Math.pow(1 - progress, 3);

            const currentVal = targetNum * eased;

            // Reconstruct the formatted string with the animated number
            if (targetText.includes('Cr')) {
                element.textContent = `₹${currentVal.toFixed(2)} Cr`;
            } else if (targetText.includes('Lakhs')) {
                element.textContent = `₹${currentVal.toFixed(2)} Lakhs`;
            } else {
                element.textContent = `₹${currentVal.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
            }

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                element.textContent = targetText;
            }
        }

        requestAnimationFrame(update);
    }

    // ---- Render prediction result ----
    function renderResult(data) {
        if (!resultContainer) return;

        const { formatted_price, inputs, saved_to_db } = data;
        const { area, bedrooms, bathrooms, location } = inputs;

        resultContainer.innerHTML = `
            <div class="result-card">
                <span class="result-icon">🏠</span>
                <span class="result-label">Estimated Property Value</span>
                <h2 class="result-value" id="result-animated">${formatted_price}</h2>
                <p class="result-subtitle">
                    ${saved_to_db ? '✓ Saved to history' : '⚠ Not saved (database unavailable)'}
                </p>
                <div class="result-details">
                    <div class="result-detail-item">
                        <div class="result-detail-label">Area</div>
                        <div class="result-detail-value">${Number(area).toLocaleString()} ft²</div>
                    </div>
                    <div class="result-detail-item">
                        <div class="result-detail-label">Bedrooms</div>
                        <div class="result-detail-value">${bedrooms}</div>
                    </div>
                    <div class="result-detail-item">
                        <div class="result-detail-label">Bathrooms</div>
                        <div class="result-detail-value">${bathrooms}</div>
                    </div>
                    <div class="result-detail-item">
                        <div class="result-detail-label">Location</div>
                        <div class="result-detail-value">${location}</div>
                    </div>
                </div>
                <button class="btn-add-compare" id="btn-add-compare">
                    <span style="margin-right: 6px;">+</span> Add to Compare
                </button>
            </div>
        `;

        // Store the prediction to be used by comparison
        currentPrediction = data;

        // Animate the price counter
        const animEl = document.getElementById('result-animated');
        if (animEl) {
            animateCounter(animEl, formatted_price);
        }

        // Smooth scroll to the result
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // Bind Add to Compare button
        const btnCompare = document.getElementById('btn-add-compare');
        if (btnCompare) {
            btnCompare.addEventListener('click', () => {
                addToCompare(currentPrediction);
            });
        }
    }

    // ---- Property Comparison Logic ----
    function saveCompareList() {
        localStorage.setItem('compareList', JSON.stringify(compareList));
    }

    function addToCompare(predictionData) {
        if (!predictionData) return;
        
        // Limit to 4 items
        if (compareList.length >= 4) {
            showToast('You can compare a maximum of 4 properties at once.', 'warning');
            return;
        }

        // Add to list
        compareList.push({
            id: Date.now().toString(),
            ...predictionData
        });
        
        saveCompareList();
        showToast('Property added to comparison.', 'success');
        renderComparisonTable();
    }
    
    // Expose for inline handlers
    window.addToCompare = addToCompare;

    window.removeFromCompare = function(id) {
        compareList = compareList.filter(item => item.id !== id);
        saveCompareList();
        renderComparisonTable();
    };

    window.addEventListener('storage', (e) => {
        if (e.key === 'compareList') {
            try {
                const stored = localStorage.getItem('compareList');
                compareList = stored ? JSON.parse(stored) : [];
                renderComparisonTable();
            } catch(err) {}
        }
    });

    function renderComparisonTable() {
        if (!comparisonContainer) return;
        
        if (compareList.length === 0) {
            comparisonContainer.innerHTML = '<p class="text-muted comparison-empty">Add properties from the results to compare them side-by-side.</p>';
            return;
        }

        let html = '';
        compareList.forEach(item => {
            const { formatted_price, inputs, predicted_price } = item;
            const { area, bedrooms, bathrooms, location } = inputs;
            
            // Calculate price per sqft
            let psqft = 0;
            if (predicted_price && area > 0) {
                psqft = Math.round(predicted_price / area);
            }
            const psqftFormatted = '₹' + psqft.toLocaleString('en-IN') + '/sq.ft';

            html += `
                <div class="compare-item">
                    <button class="compare-remove" onclick="removeFromCompare('${item.id}')" title="Remove">✕</button>
                    <div class="compare-price">${formatted_price}</div>
                    <div class="compare-metrics">
                        <div class="compare-metric"><span>Area:</span><span>${Number(area).toLocaleString()} ft²</span></div>
                        <div class="compare-metric"><span>Beds/Baths:</span><span>${bedrooms} / ${bathrooms}</span></div>
                        <div class="compare-metric"><span>Rate:</span><span>${psqftFormatted}</span></div>
                        <div class="compare-metric"><span>City:</span><span style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${location}">${location}</span></div>
                    </div>
                </div>
            `;
        });

        comparisonContainer.innerHTML = html;
    }

    // ---- Set loading state ----
    function setLoading(loading) {
        if (!submitBtn) return;

        if (loading) {
            submitBtn.disabled = true;
            submitBtn.classList.add('loading');
            if (btnText) btnText.textContent = 'Predicting...';
            if (btnSpinner) btnSpinner.style.display = 'block';
        } else {
            submitBtn.disabled = false;
            submitBtn.classList.remove('loading');
            if (btnText) btnText.textContent = 'Predict Price';
            if (btnSpinner) btnSpinner.style.display = 'none';
        }
    }

    // ---- Form submission (AJAX) ----
    if (predictionForm) {
        predictionForm.addEventListener('submit', async function (e) {
            e.preventDefault();

            if (!validateForm()) return;

            const formData = {
                area: parseFloat(document.getElementById('area').value),
                bedrooms: parseInt(document.getElementById('bedrooms').value),
                bathrooms: parseInt(document.getElementById('bathrooms').value),
                location: document.getElementById('location').value
            };

            setLoading(true);

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });

                const data = await response.json();

                if (data.success) {
                    renderResult(data);
                    if (data.fallback_used) {
                        showToast(`City not in dataset. Used an intelligent baseline estimate for ${data.inputs.location}.`, 'warning', 6000);
                    } else {
                        showToast('Prediction generated successfully!', 'success');
                    }
                } else {
                    showToast(data.error || 'Prediction failed.', 'error');
                }
            } catch (err) {
                console.error('Prediction request failed:', err);
                showToast('Network error. Please check your connection.', 'error');
            } finally {
                setLoading(false);
            }
        });
    }

    // ---- Focus animation for inputs ----
    document.querySelectorAll('.form-group input, .form-group select').forEach(el => {
        el.addEventListener('focus', () => {
            el.closest('.form-group')?.classList.add('focused');
        });
        el.addEventListener('blur', () => {
            el.closest('.form-group')?.classList.remove('focused');
        });
    });

    // ---- Keyboard shortcut: Ctrl+Enter to submit ----
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            if (predictionForm) {
                predictionForm.dispatchEvent(new Event('submit'));
            }
        }
    });

    // ---- Stagger animation for elements on page load ----
    function staggerReveal() {
        const elements = document.querySelectorAll('.stagger-item');
        elements.forEach((el, i) => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(20px)';
            el.style.transition = `opacity 0.5s ease ${i * 0.08}s, transform 0.5s ease ${i * 0.08}s`;
            requestAnimationFrame(() => {
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            });
        });
    }

    // Run on DOM ready
    staggerReveal();

})();
