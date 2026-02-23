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
    <div className="glass-panel p-6 hover:border-premium-500/50 group cursor-default relative overflow-hidden transition-all duration-500">
        {/* Animated Background Pulse */}
        <div className={cn("absolute inset-0 opacity-0 group-hover:opacity-10 transition-opacity duration-700 bg-gradient-to-br", color.replace('text-', 'from-'))} />

        <div className="flex justify-between items-start mb-6 relative z-10">
            <div className={cn("p-3 rounded-xl bg-white/5 border border-white/10 shadow-xl transition-all duration-500 group-hover:scale-110 group-hover:rotate-3", color)}>
                <Icon size={22} className="drop-shadow-[0_0_8px_currentColor]" />
            </div>
            <div className="group/info relative">
                <Info size={14} className="text-slate-600 hover:text-slate-400 transition-colors cursor-pointer" />
                <div className="absolute bottom-full right-0 mb-4 w-72 p-5 bg-[#111622]/95 border border-white/10 rounded-2xl shadow-3xl opacity-0 translate-y-2 group-hover/info:opacity-100 group-hover/info:translate-y-0 transition-all duration-300 pointer-events-none z-50 backdrop-blur-2xl">
                    <div className="flex items-center gap-2 mb-2">
                        <div className={cn("w-1.5 h-1.5 rounded-full animate-pulse", color.replace('text-', 'bg-'))} />
                        <p className="font-bold text-white uppercase tracking-[0.2em] text-[10px]">{label}</p>
                    </div>
                    <p className="text-[12px] text-slate-400 leading-relaxed font-medium">
                        {description}
                    </p>
                </div>
            </div>
        </div>

        <div className="relative z-10">
            <div className="flex items-center gap-2 mb-1">
                <h3 className="text-slate-500 text-[10px] font-bold uppercase tracking-[0.25em]">{label}</h3>
                <div className="h-[1px] flex-grow bg-white/5" />
            </div>
            <div className="flex items-baseline justify-between gap-3">
                <p className="text-4xl font-bold text-white tracking-tighter font-mono bg-clip-text text-transparent bg-gradient-to-b from-white to-white/70">
                    {value}
                </p>
                {trend && (
                    <div className={cn(
                        "px-2 py-1 rounded-lg text-[10px] font-black flex items-center gap-1 border shadow-sm",
                        trend > 0 ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20" : "bg-rose-500/10 text-rose-400 border-rose-500/20"
                    )}>
                        {trend > 0 ? <TrendingUp size={12} /> : <ShieldAlert size={12} />}
                        {Math.abs(trend)}%
                    </div>
                )}
            </div>
        </div>

        {/* Technical Data Stream Polish */}
        <div className="absolute top-0 right-0 p-1 opacity-[0.03] select-none pointer-events-none group-hover:opacity-[0.07] transition-opacity duration-700">
            <div className="text-[8px] font-mono leading-none animate-data-stream">
                {Array.from({ length: 10 }).map((_, i) => (
                    <div key={i}>{Math.random().toString(16).slice(2, 10).toUpperCase()}</div>
                ))}
            </div>
        </div>
    </div>
);

const SectionHeader = ({ title, icon: Icon, description }: any) => (
    <div className="mb-8 pl-1 relative">
        <div className="flex items-center gap-3">
            <div className="p-2 bg-premium-500/10 rounded-xl text-premium-400 border border-premium-500/10 shadow-[0_0_15px_rgba(74,99,140,0.1)]">
                <Icon size={20} className="drop-shadow-[0_0_5px_currentColor]" />
            </div>
            <h3 className="text-xl font-black tracking-tighter text-white uppercase italic">{title}</h3>
        </div>
        {description && <p className="text-slate-500 text-[11px] mt-2 font-bold uppercase tracking-widest max-w-2xl leading-relaxed opacity-60">{description}</p>}
    </div>
);

const SidebarItem = ({ label, active = false }: any) => (
    <div className={cn(
        "px-4 py-2 rounded-lg cursor-pointer transition-all text-[11px] font-bold uppercase tracking-widest",
        active ? "text-premium-400 bg-white/[0.02] shadow-[inset_0_1px_4px_rgba(0,0,0,0.4)]" : "text-slate-600 hover:text-slate-400"
    )}>
        {label}
    </div>
);

const SidebarCategory = ({ label, icon: Icon, children, active = false }: any) => (
    <div className="space-y-3">
        <div className={cn(
            "flex items-center gap-3 px-4 py-3 rounded-xl cursor-pointer transition-all duration-300 group",
            active ? "bg-premium-600/10 text-premium-400 shadow-xl" : "text-slate-500 hover:bg-white/[0.03] hover:text-slate-300"
        )}>
            <div className={cn(
                "w-8 h-8 rounded-lg flex items-center justify-center transition-all",
                active ? "bg-premium-500/20 text-premium-400" : "bg-white/5 text-slate-600 group-hover:bg-white/10"
            )}>
                <Icon size={18} />
            </div>
            <span className="hidden lg:block text-[11px] font-black tracking-[0.2em] uppercase">{label}</span>
            {active && <div className="hidden lg:block ml-auto w-1 h-4 rounded-full bg-premium-500 shadow-[0_0_12px_rgba(74,99,140,0.8)]" />}
        </div>
        {active && children && <div className="hidden lg:block pl-11 space-y-2">{children}</div>}
    </div>
);

const NavigationRail = ({ setDateRange, dateRange, showBenchmark, setShowBenchmark }: any) => (
    <div className="w-20 lg:w-72 h-screen border-r border-white/5 flex flex-col items-center py-8 lg:px-6 relative z-30 bg-[#05070a]/50 backdrop-blur-md shrink-0 sticky top-0">
        <div className="mb-12 flex items-center gap-4 lg:w-full">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-premium-400 to-premium-600 flex items-center justify-center p-0.5 shadow-2xl shadow-premium-500/20 group cursor-pointer hover:rotate-3 transition-transform">
                <div className="w-full h-full rounded-[14px] bg-[#0a0a0b] flex items-center justify-center">
                    <Zap size={22} className="text-premium-400 fill-premium-400/10 group-hover:scale-110 transition-transform" />
                </div>
            </div>
            <div className="hidden lg:block">
                <h1 className="text-sm font-black tracking-[0.3em] uppercase text-white">Alpha Engine</h1>
                <p className="text-[10px] font-bold text-premium-500 tracking-widest uppercase opacity-80">Execution Terminal v2.0</p>
            </div>
        </div>

        <div className="flex-grow space-y-6 w-full">
            <SidebarCategory label="Intelligence" icon={LayoutDashboard} active>
                <SidebarItem label="Strategy Metrics" active />
                <SidebarItem label="Alpha Factors" />
                <SidebarItem label="Risk Analysis" />
            </SidebarCategory>
            <SidebarCategory label="Operations" icon={Layers}>
                <SidebarItem label="Order Flow" />
                <SidebarItem label="Live Pipeline" />
                <SidebarItem label="Config" />
            </SidebarCategory>
        </div>

        <div className="mt-auto w-full space-y-4">
            <div className="hidden lg:block glass-panel p-4 bg-white/[0.02]">
                <div className="flex items-center justify-between mb-3">
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">System Load</span>
                    <span className="text-[9px] font-mono text-emerald-400">NORMAL</span>
                </div>
                <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden">
                    <div className="h-full w-2/3 bg-gradient-to-r from-premium-600 to-premium-400 rounded-full shadow-[0_0_10px_rgba(93,120,163,0.5)]" />
                </div>
            </div>

            <div className="flex items-center gap-3 p-3 rounded-xl hover:bg-white/5 cursor-pointer text-slate-500 hover:text-white transition-all">
                <Settings size={20} />
                <span className="hidden lg:block text-xs font-bold uppercase tracking-widest">Control Panel</span>
            </div>
        </div>
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
        <div className="min-h-screen bg-[#05070a] text-slate-200 selection:bg-premium-500/30 flex font-sans relative overflow-hidden">
            {/* Background Polish */}
            <div className="noise-filter" />
            <div className="scanline" />
            <div className="absolute top-0 left-0 w-full h-[1000px] bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-premium-900/40 via-transparent to-transparent pointer-events-none opacity-40" />

            {/* Sidebar Navigation */}
            <NavigationRail
                setDateRange={setDateRange}
                dateRange={dateRange}
                showBenchmark={showBenchmark}
                setShowBenchmark={setShowBenchmark}
            />

            {/* Main Surface */}
            <main className="flex-grow h-screen overflow-y-auto relative z-20 dashboard-scroll custom-scrollbar">
                {/* Global Header */}
                <header className="h-20 border-b border-white/5 bg-[#05070a]/40 backdrop-blur-md sticky top-0 z-40 px-8 flex items-center justify-between">
                    <div className="flex items-center gap-6">
                        <div className="flex flex-col">
                            <h2 className="text-[10px] font-black text-premium-500 uppercase tracking-[0.3em] mb-0.5">Terminal Selection</h2>
                            <div className="flex items-center gap-3">
                                <span className="text-sm font-bold text-white">SPY LONG-SHORT ALPHA v2</span>
                                <ChevronRight size={14} className="text-slate-600" />
                                <span className="text-[10px] font-mono bg-white/5 px-2 py-1 rounded border border-white/10 text-slate-400">#TRD-2024-X1</span>
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center gap-6">
                        <div className="flex flex-col items-end">
                            <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Market Status</span>
                            <div className="flex items-center gap-2">
                                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                                <span className="text-[11px] font-mono font-bold text-emerald-400 uppercase">Live Pipeline Connected</span>
                            </div>
                        </div>
                        <div className="h-8 w-[1px] bg-white/10 mx-2" />
                        <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center text-slate-400 hover:text-white transition-colors cursor-pointer group">
                            <Search size={18} className="group-hover:scale-110 transition-transform" />
                        </div>
                    </div>
                </header>

                <div className="p-8 max-w-[1600px] mx-auto space-y-12">
                    {/* Upper Viewport: Filters & Global Info */}
                    <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-8 glass-panel p-6 bg-white/[0.01]">
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-12">
                            <div className="space-y-1">
                                <p className="text-[9px] font-bold text-slate-600 uppercase tracking-[0.2em]">Strategy Mode</p>
                                <p className="text-xs font-black text-white uppercase italic">Learning-To-Rank</p>
                            </div>
                            <div className="space-y-1">
                                <p className="text-[9px] font-bold text-slate-600 uppercase tracking-[0.2em]">Asset Universe</p>
                                <p className="text-xs font-black text-white uppercase">US Equities (S&P 500)</p>
                            </div>
                            <div className="space-y-1">
                                <p className="text-[9px] font-bold text-slate-600 uppercase tracking-[0.2em]">Signal Horizon</p>
                                <p className="text-xs font-black text-white uppercase">5-Day Forward Return</p>
                            </div>
                            <div className="space-y-1">
                                <p className="text-[9px] font-bold text-slate-600 uppercase tracking-[0.2em]">Model Version</p>
                                <p className="text-xs font-black text-white uppercase flex items-center gap-2">
                                    <span className="w-2 h-2 rounded-full bg-premium-500 shadow-[0_0_8px_rgba(74,99,140,0.8)]" />
                                    LGBM-2.0.4-LND
                                </p>
                            </div>
                        </div>

                        <div className="flex items-center gap-2 bg-[#05070a] p-1.5 rounded-xl border border-white/10">
                            {["1Y", "3Y (Full)", "5Y"].map((range) => (
                                <button
                                    key={range}
                                    onClick={() => setDateRange(range)}
                                    className={cn(
                                        "px-5 py-2 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all",
                                        dateRange === range
                                            ? "bg-premium-600 text-white shadow-xl shadow-premium-500/20"
                                            : "text-slate-600 hover:text-slate-400"
                                    )}
                                >
                                    {range}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Performance Visualizer (Main View) */}
                    <section>
                        <div className="mb-10 flex items-end justify-between">
                            <div className="pl-1">
                                <h3 className="text-2xl font-black tracking-tighter text-white uppercase">Performance Intelligence</h3>
                                <p className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.25em] mt-1">Cross-sectional backtest results and risk attribution</p>
                            </div>
                            <div className="flex gap-4">
                                <button className="flex items-center gap-2 px-5 py-2.5 glass-panel bg-white/5 border-white/10 hover:border-premium-500/50 text-[10px] font-black uppercase tracking-widest text-white transition-all">
                                    <Target size={14} className="text-premium-400" />
                                    Target Portfolio
                                </button>
                                <button className="flex items-center gap-2 px-5 py-2.5 bg-premium-600 hover:bg-premium-500 text-[10px] font-black uppercase tracking-widest text-white transition-all rounded-xl shadow-lg shadow-premium-600/20 active:scale-95">
                                    <Activity size={14} />
                                    Live Research
                                </button>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                            <MetricCard
                                label="CAGR (Strategy)"
                                value={formatPercent(data.metrics.cagr)}
                                icon={TrendingUp}
                                color="text-premium-400"
                                trend={12.4}
                                description="Compound Annual Growth Rate represents the geometric mean return of the strategy over the backtest period, net of costs."
                            />
                            <MetricCard
                                label="Sharpe Ratio"
                                value={data.metrics.sharpe.toFixed(2)}
                                icon={Activity}
                                color="text-emerald-400"
                                trend={4.2}
                                description="A risk-adjusted measure that evaluates the return per unit of volatility. A value > 1.0 is considered the institutional benchmark."
                            />
                            <MetricCard
                                label="Max Drawdown"
                                value={formatPercent(data.metrics.max_drawdown)}
                                icon={ShieldAlert}
                                color="text-rose-400"
                                trend={-2.1}
                                description="The largest peak-to-trough decline observed, indicating the worst-case scenario over the historical window."
                            />
                            <MetricCard
                                label="Win Rate"
                                value={formatPercent(data.metrics.hit_rate)}
                                icon={Zap}
                                color="text-amber-400"
                                trend={0.8}
                                description="Percentage of daily cross-sectional rebalancing periods that resulted in positive strategy alpha."
                            />
                        </div>
                    </section>

                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                        {/* Equity Curve - Primary Highlight */}
                        <ChartCard
                            title="Capital Evolution"
                            subtitle="Strategy Alpha vs S&P 500 Benchmark"
                            icon={LineChart}
                            className="lg:col-span-2 shadow-2xl shadow-premium-500/5 group"
                        >
                            <div className="absolute top-4 right-4 flex items-center gap-4 z-30">
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded bg-premium-400" />
                                    <span className="text-[10px] font-bold text-slate-400 uppercase">Strategy</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded bg-[#222] border border-white/10" />
                                    <span className="text-[10px] font-bold text-slate-400 uppercase">SPY (BM)</span>
                                </div>
                            </div>

                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={filteredEquityCurve} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                                    <defs>
                                        <linearGradient id="colorStrat" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#5d78a3" stopOpacity={0.4} />
                                            <stop offset="95%" stopColor="#5d78a3" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#ffffff05" />
                                    <XAxis
                                        dataKey="date"
                                        stroke="#444"
                                        fontSize={9}
                                        tickLine={false}
                                        axisLine={false}
                                        tick={{ fontWeight: 800, style: { textTransform: 'uppercase' } }}
                                    />
                                    <YAxis
                                        stroke="#444"
                                        fontSize={9}
                                        tickLine={false}
                                        axisLine={false}
                                        tickFormatter={(val: any) => `$${val}`}
                                        tick={{ fontWeight: 800 }}
                                    />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: "rgba(10,10,12,0.95)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "16px", fontSize: "11px", backdropFilter: "blur(10px)", boxShadow: "0 20px 40px -10px rgba(0,0,0,0.5)" }}
                                        formatter={(val: any) => [`$${Number(val).toFixed(2)}`, "NAV"]}
                                        labelStyle={{ color: "#666", textTransform: "uppercase", fontWeight: 800, letterSpacing: "0.1em", fontSize: "9px", marginBottom: "4px" }}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="strategy"
                                        stroke="#5d78a3"
                                        strokeWidth={4}
                                        fill="url(#colorStrat)"
                                        animationDuration={1500}
                                        activeDot={{ r: 6, stroke: '#5d78a3', strokeWidth: 2, fill: '#fff' }}
                                    />
                                    {showBenchmark && (
                                        <Line
                                            type="monotone"
                                            dataKey="benchmark"
                                            stroke="#222"
                                            strokeDasharray="4 4"
                                            dot={false}
                                            strokeWidth={2}
                                        />
                                    )}
                                    <ReferenceLine y={1} stroke="#ffffff10" strokeWidth={1} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </ChartCard>

                        {/* Factor Intelligence Panel */}
                        <div className="space-y-8">
                            <ChartCard
                                title="Factor Hierarchy"
                                subtitle="Top 8 Features by Gain"
                                icon={PieChart}
                            >
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart layout="vertical" data={data.feature_importance.slice(0, 8)} margin={{ left: 10 }}>
                                        <XAxis type="number" hide />
                                        <YAxis
                                            type="category"
                                            dataKey="name"
                                            stroke="#444"
                                            fontSize={8}
                                            width={100}
                                            tickLine={false}
                                            axisLine={false}
                                            tick={{ fontWeight: 900, style: { textTransform: 'uppercase', letterSpacing: '0.05em' } }}
                                        />
                                        <Tooltip
                                            cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                                            contentStyle={{ backgroundColor: "#0b0f1a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px", fontSize: "10px" }}
                                        />
                                        <Bar dataKey="value" fill="#5d78a3" radius={[0, 6, 6, 0]} barSize={20} animationDuration={2000}>
                                            {data.feature_importance.map((entry: any, index: number) => (
                                                <Cell key={`cell-${index}`} fillOpacity={1 - index * 0.1} fill={index === 0 ? "#8ba8d9" : "#4a638c"} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </ChartCard>

                            <div className="glass-panel p-6 space-y-4">
                                <h4 className="text-[10px] font-black text-white uppercase tracking-[0.2em] mb-4 flex items-center gap-2">
                                    <div className="w-1 h-3 bg-premium-500 rounded-full" />
                                    Strategy Controls
                                </h4>
                                <div className="space-y-1">
                                    <FilterControl label="Benchmark Overlay" value={showBenchmark} />
                                    <FilterControl label="Volume Filter" value={true} />
                                    <FilterControl label="Rebalancing" value="DAILY" type="text" />
                                    <FilterControl label="Confidence" value="0.95" type="text" />
                                </div>
                                <button
                                    onClick={() => setShowBenchmark(!showBenchmark)}
                                    className="w-full mt-4 py-3 rounded-xl bg-white/5 border border-white/10 text-[10px] font-black text-white uppercase tracking-widest hover:bg-white/10 transition-all active:scale-95"
                                >
                                    Cycle Analytics View
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Lower Deck: Risk Attribution */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        <ChartCard
                            title="Risk Profile"
                            subtitle="Historical Drawdown Depth (%)"
                            icon={Scale}
                        >
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={data.drawdown_curve}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#ffffff05" />
                                    <XAxis dataKey="date" stroke="#444" fontSize={9} tickLine={false} axisLine={false} tick={{ fontWeight: 800, style: { textTransform: 'uppercase' } }} />
                                    <YAxis
                                        stroke="#444"
                                        fontSize={9}
                                        tickLine={false}
                                        axisLine={false}
                                        tickFormatter={(val: any) => `${(val * 100).toFixed(0)}%`}
                                        tick={{ fontWeight: 800 }}
                                    />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: "#0b0f1a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "16px", fontSize: "11px" }}
                                        formatter={(val: any) => [`${(Number(val) * 100).toFixed(2)}%`, "Drawdown"]}
                                    />
                                    <Area type="monotone" dataKey="drawdown" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.08} strokeWidth={2} animationDuration={1000} />
                                    <ReferenceLine y={-0.1} stroke="#f43f5e" strokeDasharray="5 5" strokeOpacity={0.3} label={{ value: '10% LIMIT', fill: '#f43f5e', fontSize: '9px', fontWeight: 900, position: 'insideBottomRight' }} />
                                </AreaChart>
                            </ResponsiveContainer>
                        </ChartCard>

                        <div className="glass-panel p-8 flex flex-col justify-center relative overflow-hidden group">
                            {/* Animated Background Graphics */}
                            <div className="absolute top-0 right-0 w-64 h-64 bg-premium-500/10 blur-[120px] rounded-full -translate-y-1/2 translate-x-1/2 group-hover:bg-premium-500/20 transition-colors duration-1000" />

                            <div className="relative z-10">
                                <div className="p-3 w-fit rounded-2xl bg-premium-500/10 text-premium-400 mb-6 border border-premium-500/20">
                                    <ShieldAlert size={28} />
                                </div>
                                <h3 className="text-2xl font-black text-white uppercase tracking-tighter mb-4">Risk-Adjusted Alpha</h3>
                                <p className="text-slate-400 text-sm leading-relaxed mb-8 max-w-md font-medium">
                                    The strategy v2.0 implements <span className="text-white font-bold text-xs uppercase tracking-wider">cross-sectional volatility targeting</span>. By neutralizing portfolio-wide exposure to large-cap beta, we achieve higher IC stability across market regimes.
                                </p>
                                <div className="grid grid-cols-2 gap-8">
                                    <div>
                                        <p className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">Downside Deviation</p>
                                        <p className="text-xl font-mono font-bold text-white">4.21%</p>
                                    </div>
                                    <div>
                                        <p className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1">Max Runup</p>
                                        <p className="text-xl font-mono font-bold text-emerald-400">+28.4%</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Live Ticker Footer */}
                <footer className="h-10 border-t border-white/5 bg-[#05070a] flex items-center overflow-hidden relative z-50">
                    <div className="flex items-center gap-8 px-8 animate-data-stream whitespace-nowrap opacity-40">
                        {Array.from({ length: 4 }).map((_, i) => (
                            <div key={i} className="flex gap-12 text-[9px] font-mono font-bold text-slate-500">
                                <span>AAPL +1.2%</span>
                                <span>MSFT -0.4%</span>
                                <span>NVDA +3.1%</span>
                                <span>ALPHA_SIGNAL_STRENGTH: 0.8442</span>
                                <span>EXECUTION_LATENCY: 12ms</span>
                                <span>MODEL_NDCG: 0.88</span>
                            </div>
                        ))}
                    </div>
                </footer>
            </main>
        </div>
    );
}
