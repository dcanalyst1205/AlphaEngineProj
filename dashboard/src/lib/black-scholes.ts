/**
 * Black-Scholes Option Pricing Engine
 */

export interface OptionsData {
    callPrice: number;
    putPrice: number;
    deltaCall: number;
    deltaPut: number;
    gamma: number;
    vega: number;
    thetaCall: number;
    thetaPut: number;
    rhoCall: number;
    rhoPut: number;
}

/**
 * Standard Normal Cumulative Distribution Function (CDF)
 */
function normalCDF(x: number): number {
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989423 * Math.exp(-x * x / 2);
    const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return x > 0 ? 1 - p : p;
}

/**
 * Standard Normal Probability Density Function (PDF)
 */
function normalPDF(x: number): number {
    return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/**
 * Black-Scholes formulation
 * @param S Current stock price
 * @param K Strike price
 * @param T Time to expiration (in years)
 * @param r Risk-free interest rate (annualized, e.g., 0.05 for 5%)
 * @param sigma Volatility (annualized, e.g., 0.2 for 20%)
 */
export function calculateBlackScholes(
    S: number,
    K: number,
    T: number,
    r: number,
    sigma: number
): OptionsData {
    // Avoid division by zero
    if (T <= 0) {
        const callPrice = Math.max(0, S - K);
        const putPrice = Math.max(0, K - S);
        return {
            callPrice,
            putPrice,
            deltaCall: S > K ? 1 : 0,
            deltaPut: S < K ? -1 : 0,
            gamma: 0,
            vega: 0,
            thetaCall: 0,
            thetaPut: 0,
            rhoCall: 0,
            rhoPut: 0,
        };
    }

    const d1 = (Math.log(S / K) + (r + (sigma * sigma) / 2) * T) / (sigma * Math.sqrt(T));
    const d2 = d1 - sigma * Math.sqrt(T);

    const Nd1 = normalCDF(d1);
    const Nd2 = normalCDF(d2);
    const N_d1 = normalCDF(-d1);
    const N_d2 = normalCDF(-d2);
    const pdfD1 = normalPDF(d1);

    const expRT = Math.exp(-r * T);

    const callPrice = S * Nd1 - K * expRT * Nd2;
    const putPrice = K * expRT * N_d2 - S * N_d1;

    // Greeks
    const deltaCall = Nd1;
    const deltaPut = Nd1 - 1;

    const gamma = pdfD1 / (S * sigma * Math.sqrt(T));
    const vega = S * Math.sqrt(T) * pdfD1 / 100; // Divided by 100 for 1% vol change

    const thetaCall = (-(S * pdfD1 * sigma) / (2 * Math.sqrt(T)) - r * K * expRT * Nd2) / 365;
    const thetaPut = (-(S * pdfD1 * sigma) / (2 * Math.sqrt(T)) + r * K * expRT * N_d2) / 365;

    const rhoCall = (K * T * expRT * Nd2) / 100;
    const rhoPut = (-K * T * expRT * N_d2) / 100;

    return {
        callPrice,
        putPrice,
        deltaCall,
        deltaPut,
        gamma,
        vega,
        thetaCall,
        thetaPut,
        rhoCall,
        rhoPut,
    };
}
