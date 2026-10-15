/**
 * app.js - Frontend Controller for LoanGuard AI Credit Intelligence System.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Element References
    const loanForm = document.getElementById('loanForm');
    const btnEvaluate = document.getElementById('btnEvaluate');
    const btnSpinner = document.getElementById('btnSpinner');

    // Range Sliders and Badges
    const appIncSlider = document.getElementById('applicantIncome');
    const valAppInc = document.getElementById('valAppInc');
    const coappIncSlider = document.getElementById('coapplicantIncome');
    const valCoappInc = document.getElementById('valCoappInc');
    const loanAmtSlider = document.getElementById('loanAmount');
    const valLoanAmt = document.getElementById('valLoanAmt');
    const loanTermSelect = document.getElementById('loanTerm');
    const valLoanTerm = document.getElementById('valLoanTerm');

    // Credit History Buttons
    const btnCreditGood = document.getElementById('btnCreditGood');
    const btnCreditBad = document.getElementById('btnCreditBad');
    const creditHistoryInput = document.getElementById('creditHistory');

    // Preset Buttons
    const presetButtons = document.querySelectorAll('.preset-btn');

    // Results Display Elements
    const verdictBanner = document.getElementById('verdictBanner');
    const verdictBadge = document.getElementById('verdictBadge');
    const verdictTitle = document.getElementById('verdictTitle');
    const verdictDesc = document.getElementById('verdictDesc');
    const gaugeFill = document.getElementById('gaugeFill');
    const gaugePct = document.getElementById('gaugePct');
    const resTotIncome = document.getElementById('resTotIncome');
    const resEmi = document.getElementById('resEmi');
    const resDti = document.getElementById('resDti');
    const resSurplus = document.getElementById('resSurplus');
    const factorsList = document.getElementById('factorsList');
    const probAppVal = document.getElementById('probAppVal');
    const probDefVal = document.getElementById('probDefVal');
    const probBarFill = document.getElementById('probBarFill');

    // Slider Event Listeners
    appIncSlider.addEventListener('input', (e) => {
        valAppInc.textContent = `$${parseInt(e.target.value).toLocaleString()}`;
    });

    coappIncSlider.addEventListener('input', (e) => {
        valCoappInc.textContent = `$${parseInt(e.target.value).toLocaleString()}`;
    });

    loanAmtSlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        valLoanAmt.textContent = `$${val}k ($${(val * 1000).toLocaleString()})`;
    });

    loanTermSelect.addEventListener('change', (e) => {
        const val = parseInt(e.target.value);
        valLoanTerm.textContent = `${val} mos (${(val / 12).toFixed(0)} yrs)`;
    });

    // Credit Bureau Toggle Handlers
    btnCreditGood.addEventListener('click', () => {
        btnCreditGood.classList.add('active');
        btnCreditBad.classList.remove('active');
        creditHistoryInput.value = '1.0';
    });

    btnCreditBad.addEventListener('click', () => {
        btnCreditBad.classList.add('active');
        btnCreditGood.classList.remove('active');
        creditHistoryInput.value = '0.0';
    });

    // Preset Scenarios Data
    let presetsCache = {};

    async function loadPresets() {
        try {
            const resp = await fetch('/api/presets');
            if (resp.ok) {
                presetsCache = await resp.json();
            }
        } catch (err) {
            console.warn('Could not load presets from API, using defaults:', err);
        }
    }

    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            presetButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const presetKey = btn.dataset.preset;
            if (presetsCache[presetKey]) {
                populateForm(presetsCache[presetKey].data);
                evaluateApplication();
            }
        });
    });

    function populateForm(data) {
        document.getElementById('gender').value = data.Gender || 'Male';
        document.getElementById('married').value = data.Married || 'Yes';
        document.getElementById('dependents').value = data.Dependents || '0';
        document.getElementById('education').value = data.Education || 'Graduate';
        document.getElementById('selfEmployed').value = data.Self_Employed || 'No';
        document.getElementById('propertyArea').value = data.Property_Area || 'Semiurban';

        appIncSlider.value = data.ApplicantIncome || 5000;
        valAppInc.textContent = `$${parseInt(appIncSlider.value).toLocaleString()}`;

        coappIncSlider.value = data.CoapplicantIncome || 0;
        valCoappInc.textContent = `$${parseInt(coappIncSlider.value).toLocaleString()}`;

        loanAmtSlider.value = data.LoanAmount || 120;
        valLoanAmt.textContent = `$${loanAmtSlider.value}k ($${(parseInt(loanAmtSlider.value) * 1000).toLocaleString()})`;

        loanTermSelect.value = String(data.Loan_Amount_Term || 360);
        valLoanTerm.textContent = `${loanTermSelect.value} mos (${(parseInt(loanTermSelect.value) / 12).toFixed(0)} yrs)`;

        const ch = parseFloat(data.Credit_History);
        if (ch === 1.0) {
            btnCreditGood.click();
        } else {
            btnCreditBad.click();
        }
    }

    // Form Submit / Evaluation Handler
    loanForm.addEventListener('submit', (e) => {
        e.preventDefault();
        evaluateApplication();
    });

    async function evaluateApplication() {
        btnSpinner.style.display = 'inline-block';
        btnEvaluate.disabled = true;

        const payload = {
            Gender: document.getElementById('gender').value,
            Married: document.getElementById('married').value,
            Dependents: document.getElementById('dependents').value,
            Education: document.getElementById('education').value,
            Self_Employed: document.getElementById('selfEmployed').value,
            ApplicantIncome: parseFloat(appIncSlider.value),
            CoapplicantIncome: parseFloat(coappIncSlider.value),
            LoanAmount: parseFloat(loanAmtSlider.value),
            Loan_Amount_Term: parseFloat(loanTermSelect.value),
            Credit_History: parseFloat(creditHistoryInput.value),
            Property_Area: document.getElementById('propertyArea').value
        };

        try {
            const resp = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!resp.ok) {
                throw new Error(`API error: ${resp.status}`);
            }

            const res = await resp.json();
            renderResults(res, payload);
        } catch (err) {
            console.error('Inference error:', err);
            alert(`Error assessing loan application: ${err.message}`);
        } finally {
            btnSpinner.style.display = 'none';
            btnEvaluate.disabled = false;
        }
    }

    function renderResults(res, payload) {
        const isApproved = res.Loan_Status === 'Y';
        const pDef = (res.Default_Probability * 100).toFixed(1);
        const pApp = (res.Approval_Probability * 100).toFixed(1);

        // Update Verdict Banner
        if (isApproved) {
            verdictBanner.classList.remove('rejected');
            verdictBadge.textContent = 'LOAN APPROVED';
            verdictTitle.textContent = `${res.Risk_Tier} Default Risk Tier`;
            verdictDesc.textContent = 'Applicant satisfies regulatory solvency, debt-service capacity, and credit score thresholds.';
        } else {
            verdictBanner.classList.add('rejected');
            verdictBadge.textContent = 'LOAN REJECTED';
            verdictTitle.textContent = `${res.Risk_Tier} Default Risk Tier`;
            verdictDesc.textContent = 'Application exceeds institutional risk appetite. Severe default probability detected.';
        }

        // Animate Circular SVG Gauge (circumference: 2 * PI * 50 = ~314)
        gaugePct.textContent = `${pDef}%`;
        const circumference = 314;
        const offset = circumference - (circumference * (res.Default_Probability));
        gaugeFill.style.strokeDashoffset = offset;

        // Update Financial Summary Cards
        const fin = res.Financial_Summary || {};
        resTotIncome.textContent = `$${(fin.Total_Income || (payload.ApplicantIncome + payload.CoapplicantIncome)).toLocaleString()}/mo`;
        resEmi.textContent = `$${(fin.Monthly_EMI || 0).toFixed(2)}/mo`;
        resDti.textContent = `${fin.Debt_To_Income_Pct || 0}%`;
        resSurplus.textContent = `$${(fin.Discretionary_Balance || 0).toLocaleString()}/mo`;

        // Update Probability Bar
        probAppVal.textContent = `${pApp}%`;
        probDefVal.textContent = `${pDef}%`;
        probBarFill.style.width = `${pApp}%`;

        // Render Factors Audit List
        factorsList.innerHTML = '';
        const factors = res.Key_Risk_Factors || [];

        if (isApproved) {
            const li = document.createElement('li');
            li.className = 'factor-item good';
            li.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                <span>Favorable repayment history aligns with credit underwriting benchmarks (Credit_History = ${payload.Credit_History})</span>
            `;
            factorsList.appendChild(li);

            const li2 = document.createElement('li');
            li2.className = 'factor-item good';
            li2.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
                <span>Adequate liquidity buffer: $${(fin.Discretionary_Balance || 0).toLocaleString()} net monthly surplus remaining after EMI</span>
            `;
            factorsList.appendChild(li2);
        } else {
            factors.forEach(f => {
                const li = document.createElement('li');
                li.className = 'factor-item bad';
                li.innerHTML = `
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                    <span>${f}</span>
                `;
                factorsList.appendChild(li);
            });
        }
    }

    // Initialize: Load presets and run initial evaluation
    loadPresets().then(() => {
        evaluateApplication();
    });
});
