"use client";

import React, { useState, useEffect } from "react";
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
    BarChart,
    Bar,
} from "recharts";
import {
    TrendingUp,
    BarChart3,
    ShieldAlert,
    Activity,
    ChevronRight,
    LayoutDashboard,
    PieChart
} from "lucide-react";

// Mock data structure matching the Python export
interface DashboardData {
    metrics: {
        total_return: number;
        cagr: number;
        sharpe: number;
        max_drawdown: number;
        volatility: number;
        hit_rate: number;
    };
    benchmark_metrics: {
        total_return: number;
        cagr: number;
        sharpe: number;
    };
    equity_curve: { date: string; strategy: number; benchmark: number }[];
    drawdown_curve: { date: string; drawdown: number }[];
    feature_importance: { name: string; value: number }[];
}

export default function Dashboard() {
    const [data, setData] = useState<DashboardData | null>(null);

    useEffect(() => {
        // In a real Vercel deployment, this would be fetched from a static JSON
        // exported by the Python script after each backtest.
        fetch("/data/dashboard_stats.json")
            .then((res) => res.json())
            .then((json) => setData(json))
            .catch((err) => console.error("Error loading dashboard data:", err));
    }, []);

    if (!data) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-[#0a0a0b] text-white">
                <div className="animate-pulse flex flex-col items-center">
                    <Activity className="w-12 h-12 text-premium-400 mb-4 animate-spin" />
                    <p className="text-lg font-light tracking-widest uppercase">Initialising Alpha Engine...</p>
                </div>
            </div>
        );
    }

    const kpis = [
        { label: "CAGR", value: `${(data.metrics.cagr * 100).toFixed(2)}%`, icon: TrendingUp, color: "text-emerald-400" },
        { label: "Sharpe Ratio", value: data.metrics.sharpe.toFixed(2), icon: BarChart3, color: "text-blue-400" },
        { label: "Max Drawdown", value: `${(data.metrics.max_drawdown * 100).toFixed(1)}%`, icon: ShieldAlert, color: "text-rose-400" },
        { label: "Volatility", value: `${(data.metrics.volatility * 100).toFixed(1)}%`, icon: Activity, color: "text-amber-400" },
    ];

    return (
        <main className="min-h-screen p-8 lg:p-12 space-y-8 max-w-[1600px] mx-auto">
            {/* Header */}
            <header className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border-b border-white/5 pb-8">
                <div>
                    <h1 className="text-4xl font-bold tracking-tight text-white flex items-center gap-3">
                        <div className="p-2 bg-gradient-to-br from-premium-500 to-premium-700 rounded-lg shadow-lg">
                            <LayoutDashboard size={28} />
                        </div>
                        Alpha Engine <span className="text-premium-400/50 font-light font-mono text-xl ml-2">v2.0</span>
                    </h1>
                    <p className="text-slate-400 mt-2 font-light">ML-Driven Multi-Factor Equity Backtest Analysis</p>
                </div>
                <div className="flex gap-4">
                    <div className="px-4 py-2 glass-panel flex flex-col items-end">
                        <span className="text-[10px] uppercase tracking-wider text-slate-500">Universe Size</span>
                        <span className="text-white font-mono">100 Tickers</span>
                    </div>
                    <div className="px-4 py-2 glass-panel flex flex-col items-end">
                        <span className="text-[10px] uppercase tracking-wider text-slate-500">Benchmark</span>
                        <span className="text-white font-mono">SPY (S&P 500)</span>
                    </div>
                </div>
            </header>

            {/* KPI Section */}
            <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {kpis.map((kpi, i) => (
                    <div key={i} className="glass-panel p-6 hover:border-white/10 transition-colors group">
                        <div className="flex justify-between items-start mb-4">
                            <div className={`p-2 rounded-lg bg-white/5 ${kpi.color}`}>
                                <kpi.icon size={20} />
                            </div>
                            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Live Static Data</span>
                        </div>
                        <h3 className="text-slate-400 text-sm font-medium">{kpi.label}</h3>
                        <p className="text-3xl font-bold mt-1 text-white">{kpi.value}</p>
                    </div>
                ))}
            </section>

            {/* Charts Section */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Equity Curve */}
                <section className="lg:col-span-2 glass-panel p-8">
                    <div className="flex justify-between items-center mb-8">
                        <h3 className="text-xl font-semibold flex items-center gap-2">
                            <TrendingUp size={20} className="text-premium-400" />
                            Cumulative Returns
                        </h3>
                        <div className="flex gap-4 text-xs font-mono">
                            <span className="flex items-center gap-2 text-premium-400">
                                <div className="w-3 h-3 bg-premium-400 rounded-full" /> Strategy
                            </span>
                            <span className="flex items-center gap-2 text-slate-500">
                                <div className="w-3 h-3 bg-white/20 rounded-full" /> Benchmark
                            </span>
                        </div>
                    </div>
                    <div className="chart-container">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={data.equity_curve}>
                                <defs>
                                    <linearGradient id="colorStrat" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#5d78a3" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#5d78a3" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                                <XAxis
                                    dataKey="date"
                                    stroke="#555"
                                    tick={{ fontSize: 10 }}
                                    minTickGap={100}
                                />
                                <YAxis
                                    stroke="#555"
                                    tick={{ fontSize: 10 }}
                                    tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                                />
                                <Tooltip
                                    contentStyle={{ backgroundColor: "#111", border: "1px solid #333", borderRadius: "8px" }}
                                    labelStyle={{ color: "#aaa" }}
                                    formatter={(val: number) => [`${(val * 100).toFixed(2)}%`, ""]}
                                />
                                <Area type="monotone" dataKey="strategy" stroke="#5d78a3" fillOpacity={1} fill="url(#colorStrat)" strokeWidth={2} />
                                <Line type="monotone" dataKey="benchmark" stroke="#333" strokeDasharray="5 5" dot={false} strokeWidth={1} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </section>

                {/* Feature Importance */}
                <section className="glass-panel p-8">
                    <h3 className="text-xl font-semibold mb-8 flex items-center gap-2">
                        <PieChart size={20} className="text-premium-400" />
                        Top Alpha Drivers
                    </h3>
                    <div className="chart-container">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart layout="vertical" data={data.feature_importance.slice(0, 10)}>
                                <XAxis type="number" hide />
                                <YAxis
                                    type="category"
                                    dataKey="name"
                                    stroke="#888"
                                    tick={{ fontSize: 9 }}
                                    width={100}
                                />
                                <Tooltip
                                    contentStyle={{ backgroundColor: "#111", border: "1px solid #333", borderRadius: "8px" }}
                                    cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                                />
                                <Bar dataKey="value" fill="#5d78a3" radius={[0, 4, 4, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </section>
            </div>

            {/* Drawdown Section */}
            <section className="glass-panel p-8">
                <h3 className="text-xl font-semibold mb-8 flex items-center gap-2 text-rose-400/80">
                    <ShieldAlert size={20} />
                    Underwater Analysis (Drawdown)
                </h3>
                <div className="chart-container" style={{ height: '200px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={data.drawdown_curve}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                            <XAxis dataKey="date" hide />
                            <YAxis
                                stroke="#555"
                                tick={{ fontSize: 10 }}
                                tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: "#111", border: "1px solid #333", borderRadius: "8px" }}
                                formatter={(val: number) => [`${(val * 100).toFixed(2)}%`, "Drawdown"]}
                            />
                            <Area type="monotone" dataKey="drawdown" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.1} />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </section>

            {/* Footer */}
            <footer className="pt-12 pb-8 text-center text-slate-600 text-[10px] uppercase tracking-[0.2em]">
                Design by Antigravity &bull; Alpha Engine Professional Quantitative Suite
            </footer>
        </main>
    );
}
