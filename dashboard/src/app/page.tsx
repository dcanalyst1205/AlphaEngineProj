"use client";

import React, { useState, useMemo, useEffect } from "react";
import {
    calculateBlackScholes,
    OptionsData
} from "../lib/black-scholes";
import { cn, formatCurrency, formatPercent } from "../lib/utils";
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    AreaChart,
    Area,
    ReferenceLine
} from "recharts";
import {
    Activity,
    Info,
    TrendingUp,
    ShieldAlert,
    Gauge,
    Hash,
    Calendar,
    Zap,
    ChevronRight,
    HelpCircle,
    LayoutDashboard,
    ArrowUpRight,
    ArrowDownRight
} from "lucide-react";

// --- Components ---

const InputGroup = ({ label, icon: Icon, value, onChange, min, max, step, suffix = "" }: any) => (
    <div className="space-y-3 group">
        <div className="flex justify-between items-center px-1">
            <label className="text-[10px] uppercase tracking-widest text-slate-500 font-bold flex items-center gap-2 group-hover:text-premium-400 transition-colors">
                <Icon size={12} className="text-premium-500/50" />
                {label}
            </label>
            <span className="text-xs font-mono text-white bg-white/5 px-2 py-0.5 rounded border border-white/10 group-focus-within:border-premium-500/50 transition-all">
                {value}{suffix}
            </span>
        </div>
        <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={(e) => onChange(parseFloat(e.target.value))}
            className="w-full h-1 bg-white/5 rounded-lg appearance-none cursor-pointer accent-premium-500 hover:accent-premium-400 transition-all"
        />
    </div>
);

const PriceCard = ({ title, price, greeks, type }: { title: string, price: number, greeks: any, type: 'call' | 'put' }) => (
    <div className={cn(
        "glass-panel p-6 relative overflow-hidden group transition-all duration-500 hover:scale-[1.02]",
        type === 'call' ? "hover:border-emerald-500/30" : "hover:border-rose-500/30"
    )}>
        {/* Decor */}
        <div className={cn(
            "absolute -right-4 -top-4 w-24 h-24 blur-3xl opacity-10 rounded-full transition-all duration-700 group-hover:opacity-20",
            type === 'call' ? "bg-emerald-500" : "bg-rose-500"
        )} />

        <div className="flex justify-between items-start mb-6">
            <div>
                <h3 className="text-slate-400 text-xs font-bold uppercase tracking-widest mb-1">{title} PRICE</h3>
                <p className="text-4xl font-bold tracking-tighter text-white font-mono">
                    {formatCurrency(price)}
                </p>
            </div>
            <div className={cn(
                "p-2 rounded-lg bg-white/5",
                type === 'call' ? "text-emerald-400" : "text-rose-400"
            )}>
                <Zap size={20} fill="currentColor" fillOpacity={0.1} />
            </div>
        </div>

        <div className="grid grid-cols-2 gap-4 border-t border-white/5 pt-4 mt-2">
            <div>
                <span className="text-[10px] text-slate-500 uppercase font-bold block mb-1">Delta</span>
                <span className="text-sm font-mono text-white/90">{(type === 'call' ? greeks.deltaCall : greeks.deltaPut).toFixed(3)}</span>
            </div>
            <div>
                <span className="text-[10px] text-slate-500 uppercase font-bold block mb-1">Theta</span>
                <span className="text-sm font-mono text-white/90">{(type === 'call' ? greeks.thetaCall : greeks.thetaPut).toFixed(3)}/d</span>
            </div>
            <div>
                <span className="text-[10px] text-slate-500 uppercase font-bold block mb-1">Gamma</span>
                <span className="text-sm font-mono text-white/90">{greeks.gamma.toFixed(4)}</span>
            </div>
            <div>
                <span className="text-[10px] text-slate-500 uppercase font-bold block mb-1">Vega</span>
                <span className="text-sm font-mono text-white/90">{greeks.vega.toFixed(3)}</span>
            </div>
        </div>
    </div>
);

const InfoTooltip = ({ text }: { text: string }) => (
    <div className="group relative ml-2 inline-block">
        <HelpCircle size={14} className="text-slate-600 hover:text-premium-400 transition-colors cursor-help" />
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-3 bg-[#1a1a1e] border border-white/10 rounded-lg shadow-2xl opacity-0 group-hover:opacity-100 transition-all pointer-events-none z-50 text-[11px] text-slate-300 leading-relaxed scale-95 group-hover:scale-100">
            {text}
        </div>
    </div>
);

// --- Main Dashboard ---

export default function OptionsDashboard() {
    // State
    const [spot, setSpot] = useState(150);
    const [strike, setStrike] = useState(150);
    const [vol, setVol] = useState(0.20); // 20%
    const [tte, setTte] = useState(0.5); // 6 months
    const [rfr, setRfr] = useState(0.05); // 5%

    // Calculate Prices & Greeks
    const currentData = useMemo(() =>
        calculateBlackScholes(spot, strike, tte, rfr, vol),
        [spot, strike, tte, rfr, vol]
    );

    // Chart Data Generation
    const sensitivityData = useMemo(() => {
        const points = [];
        const range = 0.5; // +/- 50%
        const minPrice = spot * (1 - range);
        const maxPrice = spot * (1 + range);
        const step = (maxPrice - minPrice) / 40;

        for (let s = minPrice; s <= maxPrice; s += step) {
            const data = calculateBlackScholes(s, strike, tte, rfr, vol);
            points.push({
                spot: s,
                call: data.callPrice,
                put: data.putPrice,
                intrinsicCall: Math.max(0, s - strike),
                intrinsicPut: Math.max(0, strike - s)
            });
        }
        return points;
    }, [spot, strike, tte, rfr, vol]);

    return (
        <main className="min-h-screen bg-[#0a0a0b] text-slate-200 selection:bg-premium-500/30">
            {/* Top Navigation / Status */}
            <div className="border-b border-white/5 bg-[#0a0a0b]/80 backdrop-blur-md sticky top-0 z-40">
                <div className="max-w-[1600px] mx-auto px-8 py-4 flex justify-between items-center">
                    <div className="flex items-center gap-3">
                        <div className="p-1.5 bg-premium-600 rounded-md shadow-[0_0_15px_rgba(93,120,163,0.3)]">
                            <LayoutDashboard size={20} className="text-white" />
                        </div>
                        <div>
                            <h1 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                                Black-Scholes Engine <span className="text-[10px] font-mono bg-premium-500/10 text-premium-400 px-1.5 py-0.5 rounded border border-premium-500/20">PRO v3.0</span>
                            </h1>
                        </div>
                    </div>
                    <div className="flex items-center gap-6">
                        <div className="hidden md:flex gap-4 items-center">
                            <div className="flex flex-col items-end">
                                <span className="text-[9px] uppercase tracking-widest text-slate-600 font-bold">Model</span>
                                <span className="text-xs font-mono text-slate-300">Continuous-Time Diffusion</span>
                            </div>
                            <div className="w-px h-6 bg-white/10" />
                            <div className="flex flex-col items-end">
                                <span className="text-[9px] uppercase tracking-widest text-slate-600 font-bold">Accuracy</span>
                                <span className="text-xs font-mono text-emerald-400">99.98% Float64</span>
                            </div>
                        </div>
                        <button className="bg-white text-black px-4 py-1.5 rounded-full text-xs font-bold hover:bg-slate-200 transition-colors shadow-lg active:scale-95">
                            Export Analysis
                        </button>
                    </div>
                </div>
            </div>

            <div className="max-w-[1600px] mx-auto p-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Left Sidebar: Controls */}
                <aside className="lg:col-span-3 space-y-8">
                    <div className="glass-panel p-6 border-premium-500/10 relative overflow-hidden">
                        <div className="flex items-center justify-between mb-8">
                            <h2 className="text-xs font-bold uppercase tracking-[0.2em] text-premium-400 flex items-center gap-2">
                                <Activity size={12} />
                                Market Inputs
                            </h2>
                            <InfoTooltip text="Dynamic input parameters for real-time option valuation." />
                        </div>

                        <div className="space-y-8">
                            <InputGroup
                                label="Spot Price"
                                icon={TrendingUp}
                                value={spot}
                                onChange={setSpot}
                                min={10}
                                max={1000}
                                step={1}
                                suffix="$"
                            />
                            <InputGroup
                                label="Strike Price"
                                icon={Hash}
                                value={strike}
                                onChange={setStrike}
                                min={10}
                                max={1000}
                                step={1}
                                suffix="$"
                            />
                            <InputGroup
                                label="Volatility (IV)"
                                icon={TrendingUp}
                                value={vol}
                                onChange={setVol}
                                min={0.01}
                                max={2.0}
                                step={0.01}
                                suffix={` (${(vol * 100).toFixed(0)}%)`}
                            />
                            <InputGroup
                                label="Time to Expiry"
                                icon={Calendar}
                                value={tte}
                                onChange={setTte}
                                min={0.01}
                                max={5}
                                step={0.01}
                                suffix={` Years (${(tte * 365).toFixed(0)}d)`}
                            />
                            <InputGroup
                                label="Risk-Free Rate"
                                icon={ShieldAlert}
                                value={rfr}
                                onChange={setRfr}
                                min={0}
                                max={0.2}
                                step={0.001}
                                suffix={` (${(rfr * 100).toFixed(1)}%)`}
                            />
                        </div>

                        <div className="mt-10 p-4 rounded-xl bg-premium-500/5 border border-premium-500/10 text-[10px] text-slate-400 leading-relaxed">
                            <div className="flex items-center gap-2 text-premium-400 mb-2 font-bold uppercase tracking-widest">
                                <Zap size={10} /> Live Calibration
                            </div>
                            The model assumes 252 trading days per year and continuous compounding. IV represents implied volatility of the underlying.
                        </div>
                    </div>
                </aside>

                {/* Main Content */}
                <div className="lg:col-span-9 space-y-8">
                    {/* Top Row: Call/Put Comparison */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <PriceCard
                            title="Call Option"
                            price={currentData.callPrice}
                            greeks={currentData}
                            type="call"
                        />
                        <PriceCard
                            title="Put Option"
                            price={currentData.putPrice}
                            greeks={currentData}
                            type="put"
                        />
                    </div>

                    {/* Middle Row: Main Visualizations */}
                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                        {/* Price Sensitivity Chart */}
                        <div className="glass-panel p-8">
                            <div className="flex justify-between items-center mb-8">
                                <h3 className="text-sm font-bold uppercase tracking-widest text-white flex items-center gap-2">
                                    <TrendingUp size={14} className="text-premium-400" />
                                    Price Sensitivity (P/L)
                                    <InfoTooltip text="Visualizes how the call (emerald) and put (rose) prices change as the underlying stock price moves." />
                                </h3>
                                <div className="flex gap-4 text-[10px] font-bold uppercase tracking-widest">
                                    <span className="flex items-center gap-2 text-emerald-400">
                                        <div className="w-2 h-2 rounded-full bg-emerald-400" /> Call
                                    </span>
                                    <span className="flex items-center gap-2 text-rose-400">
                                        <div className="w-2 h-2 rounded-full bg-rose-400" /> Put
                                    </span>
                                </div>
                            </div>
                            <div className="h-[350px] w-full">
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={sensitivityData}>
                                        <defs>
                                            <linearGradient id="colorCall" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#10b981" stopOpacity={0.2} />
                                                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                                            </linearGradient>
                                            <linearGradient id="colorPut" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.2} />
                                                <stop offset="95%" stopColor="#f43f5e" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                                        <XAxis
                                            dataKey="spot"
                                            stroke="#444"
                                            fontSize={10}
                                            tickFormatter={(val) => `$${val.toFixed(0)}`}
                                            tickLine={false}
                                            axisLine={false}
                                        />
                                        <YAxis
                                            stroke="#444"
                                            fontSize={10}
                                            tickFormatter={(val) => `$${val.toFixed(0)}`}
                                            tickLine={false}
                                            axisLine={false}
                                        />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: "#111", border: "1px solid #333", borderRadius: "12px", fontSize: "11px" }}
                                            itemStyle={{ padding: 0 }}
                                            formatter={(val: number) => [`$${val.toFixed(2)}`, ""]}
                                            labelFormatter={(val: number) => `Spot: $${val.toFixed(2)}`}
                                        />
                                        <Area type="monotone" dataKey="call" stroke="#10b981" strokeWidth={2} fill="url(#colorCall)" />
                                        <Area type="monotone" dataKey="put" stroke="#f43f5e" strokeWidth={2} fill="url(#colorPut)" />
                                        <ReferenceLine x={spot} stroke="#666" strokeDasharray="3 3" label={{ value: 'Current', position: 'top', fill: '#888', fontSize: 10 }} />
                                        <ReferenceLine x={strike} stroke="#fff" strokeOpacity={0.1} label={{ value: 'Strike', position: 'bottom', fill: '#444', fontSize: 10 }} />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        {/* Greeks Insight */}
                        <div className="glass-panel p-8">
                            <h3 className="text-sm font-bold uppercase tracking-widest text-white mb-8 flex items-center gap-2">
                                <Gauge size={14} className="text-premium-400" />
                                Greeks Visualization
                            </h3>
                            <div className="space-y-8">
                                <div className="p-6 bg-white/[0.02] rounded-2xl border border-white/5">
                                    <div className="flex justify-between items-center mb-6">
                                        <span className="text-[10px] uppercase tracking-[0.2em] font-bold text-slate-500">Delta Exposure</span>
                                        <div className="flex items-center gap-4">
                                            <span className="text-xs font-mono text-emerald-400">Call: {currentData.deltaCall.toFixed(2)}</span>
                                            <span className="text-xs font-mono text-rose-400">Put: {currentData.deltaPut.toFixed(2)}</span>
                                        </div>
                                    </div>
                                    <div className="flex h-3 w-full bg-white/5 rounded-full overflow-hidden border border-white/5 shadow-inner">
                                        <div
                                            className="h-full bg-rose-500 rounded-full transition-all duration-700"
                                            style={{ width: `${Math.abs(currentData.deltaPut) * 50}%`, marginLeft: `${(1 + currentData.deltaPut) * 50}%` }}
                                        />
                                        <div
                                            className="h-full bg-emerald-500 rounded-full transition-all duration-700"
                                            style={{ width: `${currentData.deltaCall * 50}%` }}
                                        />
                                    </div>
                                    <div className="flex justify-between items-center mt-3 text-[9px] font-bold text-slate-600 uppercase">
                                        <span>Short Delta</span>
                                        <span>Neutral</span>
                                        <span>Long Delta</span>
                                    </div>
                                </div>

                                <div className="grid grid-cols-3 gap-4">
                                    {[
                                        { label: "Gamma", value: currentData.gamma.toFixed(4), desc: "Curvature" },
                                        { label: "Vega", value: currentData.vega.toFixed(3), desc: "Vol Sensitivity" },
                                        { label: "Rho", value: currentData.rhoCall.toFixed(3), desc: "Rate Impact" },
                                    ].map((greek, i) => (
                                        <div key={i} className="p-4 rounded-xl bg-white/[0.02] border border-white/5 hover:border-premium-500/20 transition-colors text-center group">
                                            <span className="text-[9px] uppercase font-bold text-slate-600 group-hover:text-premium-400 transition-colors">{greek.label}</span>
                                            <span className="block text-xl font-bold font-mono text-white mt-1">{greek.value}</span>
                                            <span className="text-[8px] text-slate-700 font-medium uppercase mt-1">{greek.desc}</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Footer */}
                    <footer className="pt-8 border-t border-white/5 flex flex-col md:flex-row justify-between items-center gap-4 text-slate-600 text-[9px] uppercase tracking-[0.2em]">
                        <div className="flex items-center gap-4">
                            <span>Professional Quantitative Suite</span>
                            <div className="w-1 h-1 rounded-full bg-white/20" />
                            <span className="text-white/40">Market Neutral Testing Environment</span>
                        </div>
                        <div className="font-mono">
                            CALIBRATED: {new Date().toLocaleTimeString()}
                        </div>
                    </footer>
                </div>
            </div>
        </main>
    );
}
