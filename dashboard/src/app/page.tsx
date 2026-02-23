"use client";

import React, { useState, useEffect, useMemo } from "react";
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
    Cell,
    ReferenceLine
} from "recharts";
import {
    TrendingUp,
    ShieldAlert,
    Activity,
    ChevronRight,
    LayoutDashboard,
    PieChart,
    Zap,
    Scale,
    Target,
    Filter,
    ArrowUpRight,
    ArrowDownRight,
    Search,
    RefreshCcw,
    Layers,
    Info,
    Calendar,
    Settings,
    Clock,
    BarChart3
} from "lucide-react";
import { cn, formatPercent } from "../lib/utils";

// --- Types ---

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

// --- Sub-Components ---

const MetricCard = ({ label, value, icon: Icon, color, description, trend }: any) => (
    <div className="glass-panel p-6 hover:border-premium-500/30 transition-all duration-500 group cursor-default relative overflow-hidden">
        <div className="flex justify-between items-start mb-4 relative z-10">
            <div className={cn("p-2.5 rounded-xl bg-white/5 shadow-inner transition-transform duration-500 group-hover:scale-110", color)}>
                <Icon size={20} />
            </div>
            <div className="group/info relative">
                <Info size={14} className="text-slate-600 hover:text-slate-400 transition-colors" />
                <div className="absolute bottom-full right-0 mb-3 w-64 p-4 bg-[#0d0d0f] border border-white/10 rounded-xl shadow-2xl opacity-0 translate-y-2 group-hover/info:opacity-100 group-hover/info:translate-y-0 transition-all pointer-events-none z-50 text-[11px] text-slate-300 font-medium leading-relaxed backdrop-blur-xl">
                    <p className="font-bold text-white mb-1 uppercase tracking-widest text-[9px]">{label}</p>
                    {description}
                </div>
            </div>
        </div>
        <div className="relative z-10">
            <h3 className="text-slate-500 text-[10px] font-bold uppercase tracking-[0.25em] mb-1">{label}</h3>
            <div className="flex items-baseline gap-3">
                <p className="text-3xl font-bold text-white tracking-tighter font-mono">{value}</p>
                {trend && (
                    <span className={cn("text-[10px] font-bold flex items-center gap-0.5", trend > 0 ? "text-emerald-500" : "text-rose-500")}>
                        {trend > 0 ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
                        {Math.abs(trend)}%
                    </span>
                )}
            </div>
        </div>
        {/* Subtle background glow */}
        <div className={cn("absolute -right-4 -bottom-4 w-24 h-24 blur-3xl opacity-5 rounded-full transition-opacity duration-500 group-hover:opacity-10", color.replace('text-', 'bg-'))} />
    </div>
);

const SectionHeader = ({ title, icon: Icon, description }: any) => (
    <div className="mb-8 pl-1">
        <h3 className="text-lg font-bold tracking-tight text-white flex items-center gap-3">
            <div className="p-1.5 bg-premium-500/10 rounded-lg text-premium-400">
                <Icon size={18} />
            </div>
            {title}
        </h3>
        {description && <p className="text-slate-500 text-xs mt-1 font-medium">{description}</p>}
    </div>
);

const ChartCard = ({ title, icon: Icon, children, className, subtitle }: any) => (
    <div className={cn("glass-panel p-8 group", className)}>
        <div className="flex justify-between items-start mb-8">
            <div>
                <h3 className="text-sm font-bold tracking-[0.2em] text-white uppercase flex items-center gap-3 mb-1">
                    <Icon size={16} className="text-premium-400 group-hover:rotate-12 transition-transform duration-500" />
                    {title}
                </h3>
                {subtitle && <p className="text-[10px] text-slate-500 font-semibold uppercase tracking-wider">{subtitle}</p>}
            </div>
            <div className="flex gap-2">
                <button className="p-2 hover:bg-white/5 rounded-lg transition-all text-slate-600 hover:text-white border border-transparent hover:border-white/10 active:scale-95">
                    <RefreshCcw size={14} />
                </button>
            </div>
        </div>
        <div className="h-[350px] w-full relative">
            {children}
        </div>
    </div>
);

const SidebarCategory = ({ label, icon: Icon, children, active = false }: any) => (
    <div className="space-y-2">
        <div className={cn(
            "flex items-center gap-3 px-4 py-2.5 rounded-xl cursor-pointer transition-all duration-300 group",
            active ? "bg-premium-600/10 text-premium-400 shadow-[inset_0_1px_10px_rgba(0,0,0,0.2)]" : "text-slate-500 hover:bg-white/[0.03] hover:text-slate-300"
        )}>
            <Icon size={18} className={cn("transition-colors", active ? "text-premium-400" : "text-slate-600 group-hover:text-slate-400")} />
            <span className="text-xs font-bold tracking-wide uppercase">{label}</span>
            {active && <div className="ml-auto w-1.5 h-1.5 rounded-full bg-premium-500 animate-pulse shadow-[0_0_8px_rgba(74,99,140,0.8)]" />}
        </div>
        {active && children && <div className="pl-11 space-y-1">{children}</div>}
    </div>
);

const FilterControl = ({ label, value, type = "toggle" }: any) => (
    <div className="flex items-center justify-between py-1.5">
        <span className="text-[10px] text-slate-600 font-bold uppercase tracking-wider">{label}</span>
        {type === "toggle" ? (
            <div className={cn("w-7 h-4 rounded-full p-0.5 cursor-pointer transition-colors duration-300", value ? "bg-premium-500" : "bg-white/10")}>
                <div className={cn("w-3 h-3 bg-white rounded-full transition-transform duration-300 shadow-sm", value ? "translate-x-3" : "translate-x-0")} />
            </div>
        ) : (
            <span className="text-[10px] font-mono text-premium-400">{value}</span>
        )}
    </div>
);

// --- Main Dashboard ---

export default function AlphaEngineDashboard() {
    const [data, setData] = useState<DashboardData | null>(null);
    const [loading, setLoading] = useState(true);
    const [showBenchmark, setShowBenchmark] = useState(true);
    const [dateRange, setDateRange] = useState("3Y (Full)");

    useEffect(() => {
        fetch("/data/dashboard_stats.json")
            .then((res) => res.json())
            .then((json) => {
                setData(json);
                setLoading(false);
            })
            .catch((err) => {
                console.error("Error loading dashboard data:", err);
                setLoading(false);
            });
    }, []);

    const filteredEquityCurve = useMemo(() => {
        if (!data) return [];
        // Local filtering simulation
        if (dateRange === "1Y") return data.equity_curve.slice(-252);
        return data.equity_curve;
    }, [data, dateRange]);

    if (loading || !data) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-[#0a0a0b]">
                <div className="flex flex-col items-center gap-8">
                    <div className="relative">
                        <div className="w-16 h-16 border-t-2 border-r-2 border-premium-500 rounded-full animate-spin" />
                        <Activity className="absolute inset-0 m-auto w-6 h-6 text-premium-400 animate-pulse" />
                    </div>
                    <div className="text-center space-y-2">
                        <p className="text-xs font-bold tracking-[0.4em] uppercase text-white">Initialising Alpha Stream</p>
                        <p className="text-[10px] font-mono text-slate-600 uppercase">Connecting to Tier-1 Liquidity Node...</p>
                    </div>
                </div>
            </div>
        );
    }

    const kpis = [
        {
            label: "CAGR",
            value: formatPercent(data.metrics.cagr),
            icon: TrendingUp,
            color: "text-emerald-400",
            trend: 1.2,
            description: "Compound Annual Growth Rate. Represents the smoothed annual return of the strategy, accounting for compounding over the backtest period."
        },
        {
            label: "Information Ratio",
            value: data.metrics.sharpe.toFixed(2),
            icon: Scale,
            color: "text-blue-400",
            description: "A measure of risk-adjusted return. Specifically, it represents the strategy's excess return per unit of active risk (tracking error)."
        },
        {
            label: "Max Drawdown",
            value: formatPercent(data.metrics.max_drawdown),
            icon: ShieldAlert,
            color: "text-rose-400",
            description: "The peak-to-trough decline during a specific record period. It measures the theoretical maximum loss a portfolio could have suffered."
        },
        {
            label: "Alpha Hit Rate",
            value: formatPercent(data.metrics.hit_rate),
            icon: Target,
            color: "text-amber-400",
            description: "The proportion of periods where the engine successfully predicted the rank-order correctly relative to the cross-sectional mean."
        },
    ];

    return (
        <main className="min-h-screen bg-[#070708] flex selection:bg-premium-500/30">
            {/* Sidebar Navigation */}
            <nav className="w-72 border-r border-white/5 hidden xl:flex flex-col p-8 space-y-10 sticky top-0 h-screen bg-[#09090b]/50 backdrop-blur-3xl z-50">
                <div className="flex items-center gap-4 px-2">
                    <div className="w-10 h-10 bg-gradient-to-br from-premium-500 to-premium-700 rounded-xl shadow-2xl flex items-center justify-center border border-white/10 group cursor-pointer active:scale-95 transition-all">
                        <LayoutDashboard size={20} className="text-white group-hover:rotate-6 transition-transform" />
                    </div>
                    <div>
                        <h1 className="text-sm font-black tracking-widest text-white uppercase">Alpha Engine</h1>
                        <p className="text-[10px] font-bold text-premium-400/60 uppercase tracking-widest">Performance Terminal</p>
                    </div>
                </div>

                <div className="space-y-8">
                    <div className="space-y-1">
                        <p className="px-4 text-[9px] uppercase font-black text-slate-700 tracking-[0.3em] mb-4">Core Explorer</p>
                        <SidebarCategory label="Overview" icon={Activity} active>
                            <div className="space-y-3 pt-2">
                                <div onClick={() => setDateRange("3Y (Full)")} className={cn("text-[10px] font-bold uppercase tracking-wider cursor-pointer hover:text-white transition-colors flex items-center gap-2", dateRange === "3Y (Full)" ? "text-premium-400" : "text-slate-600")}>
                                    <Clock size={10} /> Full Backtest View
                                </div>
                                <div onClick={() => setDateRange("1Y")} className={cn("text-[10px] font-bold uppercase tracking-wider cursor-pointer hover:text-white transition-colors flex items-center gap-2", dateRange === "1Y" ? "text-premium-400" : "text-slate-600")}>
                                    <Calendar size={10} /> LTM Analysis
                                </div>
                            </div>
                        </SidebarCategory>
                        <SidebarCategory label="Factor Exposure" icon={Layers} />
                        <SidebarCategory label="Portfolio" icon={PieChart} />
                        <SidebarCategory label="Risk Center" icon={ShieldAlert} />
                    </div>

                    <div className="space-y-4">
                        <p className="px-4 text-[9px] uppercase font-black text-slate-700 tracking-[0.3em] mb-4">Deep Filters</p>
                        <div className="px-4 space-y-2">
                            <FilterControl label="Overlay Benchmark" value={showBenchmark} />
                            <div onClick={() => setShowBenchmark(!showBenchmark)}>
                                <FilterControl label="Logarithmic Scale" value={false} />
                                <FilterControl label="Signal Smoothing" value={true} />
                            </div>
                            <FilterControl label="Universe" value="S&P 500" type="text" />
                            <FilterControl label="Latency" value="2.4ms" type="text" />
                        </div>
                    </div>
                </div>

                <div className="mt-auto">
                    <div className="p-5 rounded-2xl bg-gradient-to-b from-white/[0.03] to-transparent border border-white/5 relative overflow-hidden group">
                        <Zap size={10} className="absolute top-4 right-4 text-premium-400 opacity-20 group-hover:opacity-100 transition-opacity" />
                        <p className="text-[10px] uppercase font-black text-slate-600 tracking-wider mb-3">Engine Status</p>
                        <div className="space-y-3">
                            <div className="flex justify-between items-center text-[10px] font-mono font-bold">
                                <span className="text-slate-400 uppercase">Training IC</span>
                                <span className="text-emerald-400">0.084</span>
                            </div>
                            <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                                <div className="w-[84%] h-full bg-emerald-500/50 rounded-full" />
                            </div>
                            <div className="flex justify-between items-center text-[10px] font-mono font-bold">
                                <span className="text-slate-400 uppercase">Valid. Stability</span>
                                <span className="text-premium-400">92%</span>
                            </div>
                            <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                                <div className="w-[92%] h-full bg-premium-500 rounded-full shadow-[0_0_8px_rgba(74,99,140,0.5)]" />
                            </div>
                        </div>
                    </div>
                </div>
            </nav>

            {/* Main Content Area */}
            <div className="flex-1 max-w-[1700px] p-8 lg:p-14 space-y-12 overflow-y-auto">
                {/* Global Header */}
                <header className="flex flex-col xl:flex-row justify-between items-start xl:items-center gap-10 border-b border-white/5 pb-10">
                    <div className="space-y-3">
                        <div className="flex items-center gap-4">
                            <span className="px-3 py-1 bg-premium-500/10 border border-premium-500/20 text-premium-400 text-[9px] font-black uppercase tracking-[0.2em] rounded-full">
                                LIVE Backtest Report
                            </span>
                            <span className="text-slate-600 text-[10px] font-bold uppercase tracking-widest flex items-center gap-1.5">
                                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                                Synchronized
                            </span>
                        </div>
                        <h2 className="text-4xl font-black tracking-tighter text-white">Alpha Engine <span className="text-slate-700 font-light italic">v2.0</span> Strategy Audit</h2>
                        <p className="text-slate-500 font-bold text-sm leading-relaxed max-w-2xl">
                            Learning-to-Rank framework evaluating cross-sectional returns via deep-factor decomposition. Visualizing alpha-decay, risk-profiles, and ML signal quality metrics.
                        </p>
                    </div>
                    <div className="flex flex-wrap gap-4">
                        <div className="p-1 glass-panel bg-white/[0.02] flex items-center">
                            <div className="px-5 py-3 border-r border-white/5">
                                <p className="text-[9px] uppercase tracking-[0.2em] text-slate-600 font-black mb-1">Lookback</p>
                                <p className="text-white font-mono text-xs font-bold">252D</p>
                            </div>
                            <div className="px-5 py-3">
                                <p className="text-[9px] uppercase tracking-[0.2em] text-slate-600 font-black mb-1">Portfolio</p>
                                <p className="text-white font-mono text-xs font-bold">Long/Short</p>
                            </div>
                        </div>
                        <button className="px-6 py-4 bg-white text-black font-black uppercase tracking-widest text-[10px] rounded-2xl hover:bg-premium-400 hover:text-white transition-all duration-300 shadow-2xl active:scale-95 flex items-center gap-3">
                            <RefreshCcw size={14} />
                            Recalibrate Model
                        </button>
                    </div>
                </header>

                {/* Primary Metrics */}
                <section>
                    <SectionHeader
                        title="Key Performance Indicators"
                        icon={BarChart3}
                        description="Real-time strategy health metrics adjusted for risk and transaction leakage."
                    />
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        {kpis.map((kpi, i) => (
                            <MetricCard key={i} {...kpi} />
                        ))}
                    </div>
                </section>

                {/* Analytical Charts */}
                <section className="space-y-12">
                    <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
                        {/* Cumulative Equity Curve */}
                        <ChartCard
                            title="Alpha Stream Propagation"
                            subtitle="Strategy vs S&P 500 Benchmark"
                            icon={TrendingUp}
                            className="xl:col-span-2"
                        >
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={filteredEquityCurve}>
                                    <defs>
                                        <linearGradient id="colorStrat" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#5d78a3" stopOpacity={0.15} />
                                            <stop offset="95%" stopColor="#5d78a3" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1c1c1e" vertical={false} />
                                    <XAxis
                                        dataKey="date"
                                        stroke="#333"
                                        fontSize={10}
                                        tickLine={false}
                                        axisLine={false}
                                        minTickGap={100}
                                        tick={{ fontWeight: 600 }}
                                    />
                                    <YAxis
                                        stroke="#333"
                                        fontSize={10}
                                        tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                                        tickLine={false}
                                        axisLine={false}
                                        tick={{ fontWeight: 600 }}
                                    />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: "rgba(10,10,12,0.95)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "16px", fontSize: "11px", backdropFilter: "blur(10px)", boxShadow: "0 20px 40px -10px rgba(0,0,0,0.5)" }}
                                        formatter={(val: any) => [`${(Number(val) * 100).toFixed(2)}%`, "Active Return"]}
                                        labelStyle={{ color: "#666", textTransform: "uppercase", fontWeight: 800, letterSpacing: "0.1em", fontSize: "9px", marginBottom: "4px" }}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="strategy"
                                        stroke="#5d78a3"
                                        strokeWidth={3}
                                        fill="url(#colorStrat)"
                                        animationDuration={1500}
                                    />
                                    {showBenchmark && (
                                        <Line
                                            type="monotone"
                                            dataKey="benchmark"
                                            stroke="#222"
                                            strokeDasharray="4 4"
                                            dot={false}
                                            strokeWidth={1.5}
                                        />
                                    )}
                                    <ReferenceLine y={0} stroke="#333" strokeWidth={1} />
                                </AreaChart>
                            </ResponsiveContainer>
                            <div className="flex justify-center gap-8 mt-6">
                                <div className="flex items-center gap-2.5">
                                    <div className="w-3 h-1 bg-premium-400 rounded-full" />
                                    <span className="text-[10px] font-black uppercase tracking-widest text-slate-400">Alpha Engine Strategy</span>
                                </div>
                                <div className="flex items-center gap-2.5">
                                    <div className="w-3 h-1 bg-white/10 border-t border-dashed rounded-full" />
                                    <span className="text-[10px] font-black uppercase tracking-widest text-slate-600">S&P 500 Index</span>
                                </div>
                            </div>
                        </ChartCard>

                        {/* Feature Importance Decomposition */}
                        <ChartCard title="Factor Contribution" subtitle="Alpha Predictor Significance" icon={Zap}>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart layout="vertical" data={data.feature_importance.slice(0, 10)} margin={{ left: 10 }}>
                                    <XAxis type="number" hide />
                                    <YAxis
                                        type="category"
                                        dataKey="name"
                                        stroke="#444"
                                        fontSize={9}
                                        width={100}
                                        tickLine={false}
                                        axisLine={false}
                                        tick={{ fontWeight: 800, textTransform: 'uppercase' }}
                                    />
                                    <Tooltip
                                        cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                                        contentStyle={{ backgroundColor: "#0a0a0c", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px", fontSize: "10px" }}
                                    />
                                    <Bar dataKey="value" fill="#5d78a3" radius={[0, 4, 4, 0]} animationDuration={2000}>
                                        {data.feature_importance.map((entry: any, index: number) => (
                                            <Cell key={`cell-${index}`} fillOpacity={1 - index * 0.08} fill={index === 0 ? "#8ba8d9" : "#4a638c"} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </ChartCard>
                    </div>

                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                        {/* Underwater Risk Analysis */}
                        <ChartCard title="Risk Exposure (Drawdown)" subtitle="Maximum peak-to-trough amplitude" icon={ShieldAlert} className="h-[300px]">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={data.drawdown_curve}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1c1c1e" vertical={false} />
                                    <XAxis dataKey="date" hide />
                                    <YAxis
                                        stroke="#333"
                                        fontSize={10}
                                        tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                                        tickLine={false}
                                        axisLine={false}
                                    />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: "#0a0a0c", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px", fontSize: "11px" }}
                                        formatter={(val: any) => [`${(Number(val) * 100).toFixed(2)}%`, "Drawdown"]}
                                    />
                                    <Area type="monotone" dataKey="drawdown" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.05} strokeWidth={1.5} animationDuration={1000} />
                                    <ReferenceLine y={-0.1} stroke="#f43f5e" strokeDasharray="5 5" strokeOpacity={0.2} label={{ value: '10% Limit', fill: '#444', fontSize: '9px', fontWeight: 800, position: 'insideBottomRight' }} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </ChartCard>

                        {/* Informational Cards */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                            <div className="glass-panel p-8 flex flex-col justify-between group hover:border-premium-500/20 transition-all">
                                <div>
                                    <div className="flex items-center gap-3 mb-6">
                                        <div className="p-2 bg-emerald-500/10 text-emerald-400 rounded-lg">
                                            <ArrowUpRight size={16} />
                                        </div>
                                        <h4 className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400">Model Stability</h4>
                                    </div>
                                    <p className="text-white text-lg font-bold tracking-tight mb-2">IC Decay Profile</p>
                                    <p className="text-slate-500 text-xs leading-relaxed font-medium">
                                        Predictive power (Rank IC) remains robust across 5-day holding periods with less than 12% decay in information coefficient.
                                    </p>
                                </div>
                                <div className="mt-8 flex items-center justify-between">
                                    <span className="text-[10px] uppercase font-black text-emerald-500 tracking-widest">Optimized for LTM</span>
                                    <ChevronRight size={14} className="text-slate-700 group-hover:text-white transition-colors" />
                                </div>
                            </div>

                            <div className="glass-panel p-8 flex flex-col justify-between group hover:border-premium-500/20 transition-all">
                                <div>
                                    <div className="flex items-center gap-3 mb-6">
                                        <div className="p-2 bg-premium-500/10 text-premium-400 rounded-lg">
                                            <Scale size={16} />
                                        </div>
                                        <h4 className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400">Risk Management</h4>
                                    </div>
                                    <p className="text-white text-lg font-bold tracking-tight mb-2">Cross-Sectional Vol</p>
                                    <p className="text-slate-500 text-xs leading-relaxed font-medium">
                                        Dynamic position sizing applied via Inverse-Variance scaling, ensuring active risk is concentrated in high-conviction signals.
                                    </p>
                                </div>
                                <div className="mt-8 flex items-center justify-between">
                                    <span className="text-[10px] uppercase font-black text-premium-400 tracking-widest">Risk Adjusted</span>
                                    <ChevronRight size={14} className="text-slate-700 group-hover:text-white transition-colors" />
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Final Professional Footer */}
                <footer className="pt-16 pb-12 border-t border-white/5 flex flex-col md:flex-row justify-between items-center gap-8">
                    <div className="flex items-center gap-6">
                        <div className="flex items-center gap-3">
                            <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
                            <span className="text-[9px] font-black uppercase tracking-[0.3em] text-slate-400">Core Engine Online</span>
                        </div>
                        <div className="w-px h-4 bg-white/10" />
                        <span className="text-slate-600 text-[9px] uppercase tracking-[0.3em] font-black">Powered by Alpha Engine ML Labs</span>
                    </div>
                    <div className="flex items-center gap-8 text-slate-700">
                        <span className="text-[10px] font-mono tracking-tighter">BUILD: 2024.02.23.REV2</span>
                        <span className="text-[10px] font-bold uppercase tracking-widest cursor-pointer hover:text-white transition-colors">Documentation</span>
                        <span className="text-[10px] font-bold uppercase tracking-widest cursor-pointer hover:text-white transition-colors">API Docs</span>
                    </div>
                </footer>
            </div>
        </main>
    );
}
