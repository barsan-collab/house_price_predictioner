/**
 * script.js — House Price Predictor v2.0
 * ─────────────────────────────────────────
 * Handles:
 *  • AJAX form submission via fetch() with JSON
 *  • Custom numeric stepper buttons (area / bedrooms / bathrooms)
 *  • Keyboard arrow-key increment/decrement for number inputs
 *  • Balcony field inclusion in prediction payload
 *  • Toast notification system
 *  • Dynamic result card rendering with animated price counter
 *  • Property comparison via localStorage
 *  • Stagger reveal on page load
 *  • Ctrl+Enter keyboard shortcut to submit
 */

(function () {
    'use strict';

    // ====================================================================
    // DOM References
    // ====================================================================
    const predictionForm     = document.getElementById('prediction-form');
    const submitBtn          = document.getElementById('submit-btn');
    const btnText            = submitBtn?.querySelector('.btn-text');
    const btnSpinner         = submitBtn?.querySelector('.btn-spinner');
    const btnIcon            = submitBtn?.querySelector('.btn-icon');
    const btnArrow           = submitBtn?.querySelector('.btn-arrow');
    const resultContainer    = document.getElementById('result-container');
    const toastContainer     = document.getElementById('toast-container');

    const areaInput          = document.getElementById('area');
    const bedroomsInput      = document.getElementById('bedrooms');
    const bathroomsInput     = document.getElementById('bathrooms');
    const locationInput      = document.getElementById('location');

    // Stepper buttons
    const areaDecBtn         = document.getElementById('area-dec');
    const areaIncBtn         = document.getElementById('area-inc');
    const bedsDecBtn         = document.getElementById('beds-dec');
    const bedsIncBtn         = document.getElementById('beds-inc');
    const bathsDecBtn        = document.getElementById('baths-dec');
    const bathsIncBtn        = document.getElementById('baths-inc');

    // Comparison
    let compareList = [];
    try {
        const stored = localStorage.getItem('compareList');
        if (stored) compareList = JSON.parse(stored);
    } catch (_) { compareList = []; }

    let currentPrediction = null;

    // ====================================================================
    // TOAST NOTIFICATION SYSTEM
    // ====================================================================
    function showToast(message, type = 'info', duration = 4000) {
        if (!toastContainer) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.setAttribute('role', 'status');

        const iconMap = { success: '✓', error: '✕', info: 'ℹ', warning: '⚠' };
        toast.innerHTML = `
            <span style="font-size:1.1rem;flex-shrink:0">${iconMap[type] || 'ℹ'}</span>
            <span>${message}</span>
        `;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('toast-exit');
            toast.addEventListener('animationend', () => toast.remove(), { once: true });
        }, duration);
    }

    // ====================================================================
    // NUMERIC STEPPER UTILITY
    // ====================================================================

    /**
     * Bind a decrement button, increment button, and a number input together.
     * @param {HTMLInputElement} input - The number input element
     * @param {HTMLButtonElement} decBtn - The minus button
     * @param {HTMLButtonElement} incBtn - The plus button
     * @param {number} step - How much to increment/decrement per click
     */
    function bindStepper(input, decBtn, incBtn, step) {
        if (!input) return;

        const getMin = () => parseFloat(input.min) || 0;
        const getMax = () => parseFloat(input.max) || Infinity;

        function updateBtnStates() {
            const val = parseFloat(input.value) || 0;
            if (decBtn) decBtn.disabled = val <= getMin();
            if (incBtn) incBtn.disabled = val >= getMax();
        }

        function nudge(delta) {
            let val = parseFloat(input.value) || 0;
            val = Math.min(getMax(), Math.max(getMin(), val + delta));
            input.value = val;
            input.dispatchEvent(new Event('input', { bubbles: true }));
            updateBtnStates();

            // Visual pulse on the input
            input.classList.add('stepper-pulse');
            input.addEventListener('animationend', () => {
                input.classList.remove('stepper-pulse');
            }, { once: true });
        }

        if (decBtn) {
            decBtn.addEventListener('click', () => nudge(-step));
            decBtn.addEventListener('mousedown', (e) => e.preventDefault()); // prevent focus steal
        }

        if (incBtn) {
            incBtn.addEventListener('click', () => nudge(+step));
            incBtn.addEventListener('mousedown', (e) => e.preventDefault());
        }

        // Keyboard arrow support: ↑ increments, ↓ decrements
        input.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowUp') {
                e.preventDefault();
                nudge(+step);
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                nudge(-step);
            }
        });

        // Clamp value on blur
        input.addEventListener('blur', () => {
            let val = parseFloat(input.value);
            if (isNaN(val)) return;
            val = Math.min(getMax(), Math.max(getMin(), val));
            input.value = val;
            updateBtnStates();
        });

        input.addEventListener('input', updateBtnStates);

        // Initial state
        updateBtnStates();
    }

    // Bind the three steppers
    bindStepper(areaInput,      areaDecBtn,  areaIncBtn,  50);   // area: ±50 sq.ft
    bindStepper(bedroomsInput,  bedsDecBtn,  bedsIncBtn,  1);    // bedrooms: ±1
    bindStepper(bathroomsInput, bathsDecBtn, bathsIncBtn, 1);    // bathrooms: ±1

    // ====================================================================
    // AREA HINT (square meters conversion)
    // ====================================================================
    const areaHint = document.getElementById('area-hint');
    if (areaInput && areaHint) {
        areaInput.addEventListener('input', () => {
            const val = parseFloat(areaInput.value);
            if (!isNaN(val) && val > 0) {
                const sqm = (val * 0.0929).toFixed(1);
                areaHint.textContent = `≈ ${sqm} sq. meters`;
                areaHint.style.color = 'var(--accent-cyan)';
            } else {
                areaHint.textContent = '';
            }
        });
    }

    // ====================================================================
    // STEP DOTS — activate as user interacts with each step
    // ====================================================================
    const stepDots = document.querySelectorAll('.step-dot');

    function activateStepDot(index) {
        stepDots.forEach((dot, i) => {
            dot.classList.toggle('active', i <= index);
        });
    }

    // Wire up focus events to illuminate step dots progressively
    const stepIds = ['location', 'area', 'bedrooms', 'bathrooms', 'balcony-yes', 'submit-btn'];
    stepIds.forEach((id, i) => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('focus', () => activateStepDot(i));
            el.addEventListener('click', () => activateStepDot(i));
        }
    });

    // ====================================================================
    // FORM VALIDATION
    // ====================================================================
    function validateForm() {
        const area     = parseFloat(areaInput?.value);
        const bedrooms = parseInt(bedroomsInput?.value, 10);
        const baths    = parseInt(bathroomsInput?.value, 10);
        const location = locationInput?.value?.trim();

        if (!location) {
            showToast('Please select a location from the map or type a city name.', 'error');
            locationInput?.focus();
            return false;
        }

        if (!area || area <= 0) {
            showToast('Please enter a valid total area (sq. ft.).', 'error');
            areaInput?.focus();
            return false;
        }

        if (area > 1_000_000) {
            showToast('Area value seems unrealistically large. Please double-check.', 'error');
            areaInput?.focus();
            return false;
        }

        if (!bedrooms || bedrooms < 1) {
            showToast('Bedrooms must be at least 1.', 'error');
            bedroomsInput?.focus();
            return false;
        }

        if (!baths || baths < 1) {
            showToast('Bathrooms must be at least 1.', 'error');
            bathroomsInput?.focus();
            return false;
        }

        return true;
    }

    // ====================================================================
    // ANIMATED PRICE COUNTER
    // ====================================================================
    function animateCounter(element, targetText) {
        const numStr  = targetText.replace(/[^\d.]/g, '');
        const target  = parseFloat(numStr);

        if (isNaN(target) || target === 0) {
            element.textContent = targetText;
            return;
        }

        const duration  = 1400;
        const startTime = performance.now();

        function update(now) {
            const elapsed  = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased    = 1 - Math.pow(1 - progress, 4); // ease-out quartic
            const current  = target * eased;

            if (targetText.includes('Cr')) {
                element.textContent = `₹${current.toFixed(2)} Cr`;
            } else if (targetText.includes('Lakhs')) {
                element.textContent = `₹${current.toFixed(2)} Lakhs`;
            } else {
                element.textContent = `₹${current.toLocaleString('en-IN', {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                })}`;
            }

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                element.textContent = targetText;
            }
        }

        requestAnimationFrame(update);
    }

    // ====================================================================
    // RENDER PREDICTION RESULT CARD
    // ====================================================================
    function renderResult(data) {
        if (!resultContainer) return;

        const { formatted_price, inputs, saved_to_db } = data;
        const { area, bedrooms, bathrooms, location } = inputs;
        const balcony = inputs.balcony || 'No';

        resultContainer.innerHTML = `
            <div class="result-card">
                <span class="result-icon">🏡</span>
                <span class="result-label">Estimated Market Value</span>
                <h2 class="result-value" id="result-animated">${formatted_price}</h2>
                <p class="result-subtitle">
                    ${saved_to_db
                        ? '<span style="color:var(--accent-emerald)">✓ Saved to history</span>'
                        : '<span style="color:var(--accent-amber)">⚠ Not saved (database unavailable)</span>'
                    }
                </p>
                <div class="result-details">
                    <div class="result-detail-item">
                        <div class="result-detail-label">Location</div>
                        <div class="result-detail-value" style="font-family:var(--font-sans);font-size:0.88rem">${location}</div>
                    </div>
                    <div class="result-detail-item">
                        <div class="result-detail-label">Area</div>
                        <div class="result-detail-value">${Number(area).toLocaleString()} ft²</div>
                    </div>
                    <div class="result-detail-item">
                        <div class="result-detail-label">Bedrooms</div>
                        <div class="result-detail-value">${bedrooms} BHK</div>
                    </div>
                    <div class="result-detail-item">
                        <div class="result-detail-label">Bathrooms</div>
                        <div class="result-detail-value">${bathrooms}</div>
                    </div>
                    <div class="result-detail-item">
                        <div class="result-detail-label">Balcony</div>
                        <div class="result-detail-value" style="text-transform:capitalize">${balcony}</div>
                    </div>
                    <div class="result-detail-item">
                        <div class="result-detail-label">Rate / sq.ft</div>
                        <div class="result-detail-value">₹${data.predicted_price > 0 && area > 0
                            ? Math.round(data.predicted_price / area).toLocaleString('en-IN')
                            : '—'
                        }</div>
                    </div>
                </div>
                <button class="btn-add-compare" id="btn-add-compare" type="button">
                    <span aria-hidden="true">⊕</span> Add to Comparison
                </button>
            </div>
        `;

        // Store for comparison
        currentPrediction = data;

        // Animate price counter
        const animEl = document.getElementById('result-animated');
        if (animEl) animateCounter(animEl, formatted_price);

        // Smooth-scroll to result
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Wire up "Add to Compare" button
        const compareBtn = document.getElementById('btn-add-compare');
        if (compareBtn) {
            compareBtn.addEventListener('click', () => addToCompare(currentPrediction));
        }

        // Light up all step dots
        activateStepDot(5);
    }

    // ====================================================================
    // COMPARISON LOGIC (localStorage-based)
    // ====================================================================
    function saveCompareList() {
        localStorage.setItem('compareList', JSON.stringify(compareList));
    }

    function addToCompare(predictionData) {
        if (!predictionData) return;

        if (compareList.length >= 4) {
            showToast('Maximum 4 properties can be compared at once.', 'warning');
            return;
        }

        compareList.push({
            id: Date.now().toString(),
            ...predictionData
        });
        saveCompareList();
        showToast('Property added to comparison list! Visit the Compare page.', 'success');
    }

    // Expose globally for inline handler and compare.html
    window.addToCompare    = addToCompare;
    window.removeFromCompare = function (id) {
        compareList = compareList.filter(item => item.id !== id);
        saveCompareList();
    };

    // Cross-tab sync
    window.addEventListener('storage', (e) => {
        if (e.key === 'compareList') {
            try {
                const stored = localStorage.getItem('compareList');
                compareList  = stored ? JSON.parse(stored) : [];
            } catch (_) {}
        }
    });

    // ====================================================================
    // LOADING STATE
    // ====================================================================
    function setLoading(isLoading) {
        if (!submitBtn) return;
        submitBtn.disabled = isLoading;
        submitBtn.classList.toggle('loading', isLoading);
        if (btnText)    btnText.textContent  = isLoading ? 'Predicting…' : 'Predict Price';
        if (btnIcon)    btnIcon.style.display = isLoading ? 'none' : '';
        if (btnArrow)   btnArrow.style.display = isLoading ? 'none' : '';
    }

    // ====================================================================
    // FORM SUBMISSION — AJAX (fetch + JSON API)
    // ====================================================================
    if (predictionForm) {
        predictionForm.addEventListener('submit', async function (e) {
            e.preventDefault();

            if (!validateForm()) return;

            // Read balcony (radio group — default to 'no' if nothing checked)
            const balconyChecked = predictionForm.querySelector('input[name="balcony"]:checked');
            const balcony = balconyChecked ? balconyChecked.value : 'no';

            const formData = {
                area:      parseFloat(areaInput.value),
                bedrooms:  parseInt(bedroomsInput.value, 10),
                bathrooms: parseInt(bathroomsInput.value, 10),
                location:  locationInput.value.trim(),
                balcony                               // included for extensibility
            };

            setLoading(true);

            try {
                const response = await fetch('/api/predict', {
                    method:  'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body:    JSON.stringify(formData)
                });

                const data = await response.json();

                if (data.success) {
                    // Attach balcony to inputs so result card can display it
                    data.inputs.balcony = balcony;
                    renderResult(data);

                    if (data.fallback_used) {
                        showToast(
                            `"${formData.location}" not in training dataset. ` +
                            'Using an intelligent baseline estimate.',
                            'warning',
                            6000
                        );
                    } else {
                        showToast('Valuation generated successfully! 🏡', 'success');
                    }
                } else {
                    showToast(data.error || 'Prediction failed. Please try again.', 'error');
                }

            } catch (err) {
                console.error('[Predictor] Fetch error:', err);
                showToast('Network error. Please check your connection and try again.', 'error');
            } finally {
                setLoading(false);
            }
        });
    }

    // ====================================================================
    // KEYBOARD SHORTCUT — Ctrl+Enter (or ⌘+Enter on Mac) to submit
    // ====================================================================
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            if (predictionForm) {
                e.preventDefault();
                predictionForm.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
            }
        }
    });

    // ====================================================================
    // STAGGER REVEAL — animate elements on page load
    // ====================================================================
    function staggerReveal() {
        const items = document.querySelectorAll('.stagger-item');
        items.forEach((el, i) => {
            el.style.transitionDelay = `${i * 0.07}s`;
            // Use rAF to let styles paint before toggling
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    el.classList.add('revealed');
                });
            });
        });
    }

    staggerReveal();

    // ====================================================================
    // FOCUS ANIMATION — subtle glow on form group focus
    // ====================================================================
    document.querySelectorAll('.form-step').forEach(step => {
        const focusables = step.querySelectorAll('input, button, select, textarea');
        focusables.forEach(el => {
            el.addEventListener('focus', () => step.classList.add('step-focused'));
            el.addEventListener('blur',  () => step.classList.remove('step-focused'));
        });
    });

})();
