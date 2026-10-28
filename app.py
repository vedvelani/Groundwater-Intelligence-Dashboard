    """
    Explainable ML for Groundwater Quality Classification
    Dashboard with Basic + Advanced Data Structures.
    Run: streamlit run app.py
    """
    import io, heapq, json, math
    from collections import Counter, deque, defaultdict
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st


    st.set_page_config(
        page_title="Groundwater Intelligence",
        page_icon="💧",
        layout="wide"
    )

    NAVY = "#1B3A5B"

    st.markdown(f"""
    <style>
    /* ===== Main Dashboard ===== */
    .stApp {{
        background: #f5f8fc;
    }}

    h1, h2, h3 {{
        color: {NAVY};
    }}

    div[data-testid="stMetric"] {{
        background: white;
        border: 1px solid #e4ebf3;
        padding: 16px;
        border-radius: 12px;
    }}

    div[data-testid="stDataFrame"] {{
        background: white;
    }}

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #1B3A5B, #173553);
    }}

    section[data-testid="stSidebar"] * {{
        color: white;
    }}

    section[data-testid="stSidebar"] > div {{
        padding: 1.4rem 1rem;
    }}

    /* ===== CSV Upload Label ===== */
    section[data-testid="stSidebar"]
    div[data-testid="stFileUploader"] label {{
        color: #FFFFFF !important;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 10px;
    }}

    /* ===== Upload Dropzone ===== */
    section[data-testid="stSidebar"]
    div[data-testid="stFileUploader"] section {{
        background: #F7FAFE !important;
        border: 2px dashed #8DBCE8 !important;
        border-radius: 16px !important;
        padding: 20px 12px !important;
        min-height: 175px;
        transition: all 0.2s ease;
    }}

    /* Hover Effect */
    section[data-testid="stSidebar"]
    div[data-testid="stFileUploader"] section:hover {{
        background: #EDF5FC !important;
        border-color: #FFFFFF !important;
    }}

    /* ===== Browse Files Button ===== */
    section[data-testid="stSidebar"]
    div[data-testid="stFileUploader"] button {{
        background: #1B3A5B !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 9px 20px !important;
        font-weight: 600 !important;
        transition: background 0.2s ease;
    }}

    section[data-testid="stSidebar"]
    div[data-testid="stFileUploader"] button:hover {{
        background: #2D5A83 !important;
        color: #FFFFFF !important;
    }}

    /* ===== Upload Instructions ===== */
    section[data-testid="stSidebar"]
    div[data-testid="stFileUploader"] small {{
        color: #526B85 !important;
    }}

    /* ===== Uploaded File Row ===== */
    section[data-testid="stSidebar"]
    div[data-testid="stFileUploader"]
    [data-testid="stFileUploaderFile"] {{
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid #42617F;
        border-radius: 12px;
        padding: 10px;
        margin-top: 10px;
    }}

    </style>
    """, unsafe_allow_html=True)



    QUALITY_ORDER=["Excellent","Good","Poor","Very Poor","Unsuitable"]
    QUALITY_COLORS={"Excellent":"#2a9d8f","Good":"#90be6d","Poor":"#f9c74f","Very Poor":"#f9844a","Unsuitable":"#d62828"}

    # -------------------- DATA STRUCTURES --------------------
    class LinkedList:
        """Singly linked list used to keep a short recent-search history."""
        class Node:
            def __init__(self, value): self.value, self.next=value, None
        def __init__(self): self.head=None; self.size=0
        def append(self, value):
            node=self.Node(value)
            if self.head is None: self.head=node
            else:
                cur=self.head
                while cur.next: cur=cur.next
                cur.next=node
            self.size+=1
        def values(self, limit=10):
            out=[]; cur=self.head
            while cur and len(out)<limit: out.append(cur.value); cur=cur.next
            return out

    class BSTNode:
        def __init__(self,key,row): self.key=key; self.rows=[row]; self.left=None; self.right=None
    class BinarySearchTree:
        """BST indexes numeric values; in-order traversal returns sorted records."""
        def __init__(self): self.root=None; self.count=0
        def insert(self,key,row):
            if pd.isna(key): return
            key=float(key)
            if self.root is None: self.root=BSTNode(key,row); self.count+=1; return
            cur=self.root
            while True:
                if key==cur.key: cur.rows.append(row); return
                if key<cur.key:
                    if cur.left is None: cur.left=BSTNode(key,row); self.count+=1; return
                    cur=cur.left
                else:
                    if cur.right is None: cur.right=BSTNode(key,row); self.count+=1; return
                    cur=cur.right
        def inorder(self):
            out=[]; stack=[]; cur=self.root
            while cur or stack:
                while cur: stack.append(cur); cur=cur.left
                cur=stack.pop(); out.extend(cur.rows); cur=cur.right
            return out

    def binary_search(sorted_values, target):
        lo,hi=0,len(sorted_values)-1
        while lo<=hi:
            mid=(lo+hi)//2
            if sorted_values[mid]==target: return mid
            if sorted_values[mid]<target: lo=mid+1
            else: hi=mid-1
        return -1

    def classify_wqi(v):
        # Conventional WQI bands used only when a WQI column exists and no class label is present.
        if pd.isna(v): return "Unclassified"
        if v<=25: return "Excellent"
        if v<=50: return "Good"
        if v<=75: return "Poor"
        if v<=100: return "Very Poor"
        return "Unsuitable"

    def normalize(s): return "".join(ch.lower() for ch in str(s) if ch.isalnum())

    def find_col(df, candidates):
        lookup={normalize(c):c for c in df.columns}
        for cand in candidates:
            if normalize(cand) in lookup: return lookup[normalize(cand)]
        for c in df.columns:
            if any(normalize(x) in normalize(c) for x in candidates): return c
        return None

    def clean_numeric(df):
        out=df.copy()
        for c in out.columns:
            if out[c].dtype=="object":
                # Convert columns that are mostly numeric, preserving location/category fields.
                converted=pd.to_numeric(out[c].astype(str).str.replace(",","",regex=False),errors="coerce")
                if converted.notna().mean()>=0.65: out[c]=converted
        return out

    # -------------------- HEADER + DATA LOADING --------------------
    st.title("💧 Groundwater Intelligence Dashboard")
    st.caption("Explainable ML for Groundwater Quality Classification • Basic and Advanced Data Structures")


    with st.sidebar:
        st.markdown("""
            <h3 style="
                color: white;
                margin-bottom: 20px;
                font-size: 22px;
            ">
                💧 Dashboard Controls
            </h3>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload cleaned CSV dataset",
            type=["csv"],
            help="Upload your cleaned groundwater dataset."
        )

        if uploaded is not None:
            st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.08);
                    border: 1px solid #42617F;
                    border-radius: 12px;
                    padding: 12px;
                    margin-top: 8px;
                    color: white;
                ">
                    <div style="font-weight: 600;
                                overflow-wrap: anywhere;">
                        ✅ {uploaded.name}
                    </div>
                    <div style="color: #D5E3F0;
                                font-size: 13px;
                                margin-top: 5px;">
                        {uploaded.size / (1024 * 1024):.2f} MB
                    </div>
                </div>
            """, unsafe_allow_html=True)

        demo = st.toggle(
            "Use built-in demo data if no file is uploaded",
            value=True
        )

        st.divider()

        st.markdown("**🧩 Data-structure modules**")
        st.markdown("""
        - 📋 Basic: List, stack, queue, hash map
        - 🌳 Advanced: BST, heap, graph
        - 📊 Analytics: WQI, filters, exports
        """)

    @st.cache_data(show_spinner=False)
    def load_csv(file_bytes):
        return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)

    def make_demo():
        rng=np.random.default_rng(42); n=80936
        states=np.array(["Gujarat","Rajasthan","Maharashtra","Madhya Pradesh","Andhra Pradesh","Chhattisgarh"])
        ph=np.clip(rng.normal(7.4,0.65,n),5.5,9.8)
        mg=np.clip(rng.gamma(2.2,18,n),1,150)
        hardness=np.clip(mg*2.2+rng.normal(35,25,n),10,500)
        sodium=np.clip(rng.gamma(2,22,n),1,250)
        chloride=np.clip(sodium*1.2+rng.normal(20,25,n),2,400)
        alkalinity=np.clip(rng.normal(180,70,n),20,600)
        carbonate=np.clip(rng.gamma(1.5,5,n),0,100)
        wqi=np.clip(18+np.abs(ph-7)*18+mg*.22+hardness*.035+sodium*.08+chloride*.04+rng.normal(0,7,n),5,180)
        d=pd.DataFrame({"State":rng.choice(states,n),"District":[f"District {x:02d}" for x in rng.integers(1,26,n)],
        "Potential of Hydrogen (pH)":ph,"Magnesium":mg,"Total Hardness":hardness,"Sodium":sodium,
        "Chloride":chloride,"Total Alkalinity":alkalinity,"Carbonate":carbonate,"wqi_score":wqi})
        d["wqi_class"]=d["wqi_score"].map(classify_wqi)
        return d

    if uploaded:
        try: raw=load_csv(uploaded.getvalue())
        except Exception as e: st.error(f"Could not read CSV: {e}"); st.stop()
    elif demo:
        raw=make_demo()
        st.info("Demo mode is active. Upload your actual dataset from the sidebar for project results; demo values are synthetic.")
    else:
        st.warning("Upload a CSV or enable demo data in the sidebar.")
        st.stop()

    df=clean_numeric(raw)
    df.columns=[str(c).strip() for c in df.columns]
    state_col=find_col(df,["State"])
    district_col=find_col(df,["District"])
    wqi_col=find_col(df,["wqi_score","WQI score","Water Quality Index"])
    label_col=find_col(df,["target_label","wqi_class","quality_class","water_quality_class","class","label"])
    if label_col is None and wqi_col:
        df["Dashboard Quality Class"]=df[wqi_col].map(classify_wqi); label_col="Dashboard Quality Class"
    elif label_col is None:
        st.warning("No quality-class label or WQI score was found. You can still explore numeric parameters, but class charts will be unavailable.")

    numeric_cols=df.select_dtypes(include=np.number).columns.tolist()
    chemical_hints = [
        "potential_of_hydrogen_ph",
        "ph",
        "magnesium",
        "carbonate",
        "total_hardness",
        "sodium",
        "chloride",
        "total_alkalinity",
    ]

    chemical_cols = [
        c for c in numeric_cols
        if any(normalize(hint) in normalize(c) for hint in chemical_hints)
    ]
    if not chemical_cols: chemical_cols=[c for c in numeric_cols if c!=wqi_col]
    if not chemical_cols: st.error("No numeric columns detected. Check the CSV."); st.stop()

    # -------------------- FILTERS --------------------
    with st.sidebar:
        st.subheader("Filter records")
        filtered=df.copy()
        if state_col:
            state_options=sorted(filtered[state_col].dropna().astype(str).unique())
            selected_states=st.multiselect("State",state_options,default=state_options)
            if selected_states: filtered=filtered[filtered[state_col].astype(str).isin(selected_states)]
        if label_col:
            class_options=sorted(filtered[label_col].dropna().astype(str).unique())
            selected_classes=st.multiselect("Quality class",class_options,default=class_options)
            if selected_classes: filtered=filtered[filtered[label_col].astype(str).isin(selected_classes)]
        parameter=st.selectbox("Numeric parameter",chemical_cols,index=0)
        st.caption(f"Rows after filters: {len(filtered):,}")

    # -------------------- KPI CARDS --------------------
    total=len(filtered); missing=int(filtered.isna().sum().sum())
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Records",f"{total:,}")
    c2.metric("Numeric parameters",f"{len(numeric_cols):,}")
    c3.metric("Missing cells",f"{missing:,}")
    c4.metric("States covered",filtered[state_col].nunique() if state_col else "N/A")

    tab1,tab2,tab3,tab4,tab5=st.tabs(["📊 Overview","🧪 Water Quality","🧱 Basic Data Structures","🌳 Advanced Data Structures","⬇️ Data & Export"])

    with tab1:
        left,right=st.columns(2)
        with left:
            if label_col and filtered[label_col].notna().any():
                counts=filtered[label_col].astype(str).value_counts().rename_axis("Quality class").reset_index(name="Records")
                st.plotly_chart(px.bar(counts,x="Quality class",y="Records",color="Quality class",title="Records by quality class",
                color_discrete_map=QUALITY_COLORS),use_container_width=True)
            elif state_col:
                counts=filtered[state_col].astype(str).value_counts().rename_axis("State").reset_index(name="Records")
                st.plotly_chart(px.bar(counts,x="State",y="Records",title="Records by state"),use_container_width=True)
        with right:
            if state_col:
                state_counts=filtered[state_col].astype(str).value_counts().rename_axis("State").reset_index(name="Records")
                st.plotly_chart(px.pie(state_counts,names="State",values="Records",hole=.45,title="Geographic distribution"),use_container_width=True)
            else:
                st.plotly_chart(px.histogram(filtered,x=parameter,title=f"Distribution: {parameter}"),use_container_width=True)
        
    # Distribution by quality class — improved visibility
    chart_data = filtered.dropna(subset=[parameter]).copy()

    if not chart_data.empty:
        # Limit the displayed range to the 99th percentile
        upper_limit = chart_data[parameter].quantile(0.99)

        # Exclude extreme outliers from this visualization only
        plot_data = chart_data[chart_data[parameter] <= upper_limit]

        fig = px.histogram(
            plot_data,
            x=parameter,
            color=label_col if label_col else None,
            marginal="box",
            nbins=60,
            title=f"{parameter}: Distribution by Quality Class (99th Percentile View)",
            color_discrete_map=QUALITY_COLORS
        )

        fig.update_layout(
            height=500,
            bargap=0.05,
            xaxis_title=parameter,
            yaxis_title="Number of Records",
            legend_title_text="Quality Class",
            template="plotly_white"
        )

        fig.update_xaxes(
            showgrid=True,
            zeroline=False
        )

        fig.update_yaxes(
            showgrid=True,
            rangemode="tozero"
        )

        st.plotly_chart(fig, use_container_width=True)

        excluded = len(chart_data) - len(plot_data)
        st.caption(
            f"Showing {len(plot_data):,} records up to the "
            f"99th percentile ({upper_limit:,.2f}). "
            f"{excluded:,} extreme observations are excluded from "
            "this chart only; the original dataset remains unchanged."
        )
    else:
        st.warning("No valid numeric observations are available for this chart.")

        if state_col:
            summary=filtered.groupby(state_col)[chemical_cols].mean(numeric_only=True).round(2).reset_index()
            st.subheader("Average chemical parameters by state")
            st.dataframe(summary,use_container_width=True,hide_index=True)

    with tab2:
        st.subheader("Parameter explorer")
        p1,p2=st.columns(2)
        with p1:
            
            # Parameter distribution — zoomed view
            parameter_data = filtered.dropna(subset=[parameter]).copy()

            if not parameter_data.empty:
                q99 = parameter_data[parameter].quantile(0.99)
                parameter_plot = parameter_data[
                    parameter_data[parameter] <= q99
                ]

                fig = px.histogram(
                    parameter_plot,
                    x=parameter,
                    marginal="box",
                    nbins=60,
                    title=f"{parameter} Distribution (99th Percentile View)"
                )

                fig.update_layout(
                    height=450,
                    bargap=0.05,
                    xaxis_title=parameter,
                    yaxis_title="Number of Records",
                    template="plotly_white"
                )

                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Extreme values above the 99th percentile are hidden "
                    "from this visualization only."
                )
            else:
                st.warning("No valid observations available.")

        with p2:
            if label_col:
                
                # Box plot across quality classes — handle extreme outliers
                box_data = filtered.dropna(subset=[parameter]).copy()

                if not box_data.empty:
                    upper_limit = box_data[parameter].quantile(0.99)

                    # Exclude extreme values from this plot only
                    box_plot_data = box_data[
                        box_data[parameter] <= upper_limit
                    ]

                    fig = px.box(
                        box_plot_data,
                        x=label_col,
                        y=parameter,
                        color=label_col,
                        category_orders={label_col: QUALITY_ORDER},
                        color_discrete_map=QUALITY_COLORS,
                        points=False,
                        title=f"{parameter} Across Quality Classes (99th Percentile View)"
                    )

                    fig.update_layout(
                        height=500,
                        template="plotly_white",
                        xaxis_title="Water Quality Class",
                        yaxis_title=parameter,
                        showlegend=False
                    )

                    fig.update_yaxes(
                        rangemode="tozero",
                        showgrid=True
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.caption(
                        f"Displaying values up to {upper_limit:,.2f}. "
                        f"{len(box_data) - len(box_plot_data):,} extreme "
                        "observations are excluded from this plot only."
                    )
                else:
                    st.warning("No valid values available for the box plot.")

            else:
                st.plotly_chart(px.box(filtered,y=parameter,title=f"{parameter} spread"),use_container_width=True)
        corr=filtered[chemical_cols].corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr,text_auto=".2f",aspect="auto",title="Chemical parameter correlation"),use_container_width=True)
        if wqi_col:
            st.metric("Mean WQI score",f"{filtered[wqi_col].mean():.2f}" if filtered[wqi_col].notna().any() else "N/A")

            # WQI score distribution — zoom in on the main data range
            wqi_data = filtered.dropna(subset=[wqi_col]).copy()

            if not wqi_data.empty:
                # Calculate the 99th percentile
                wqi_upper = wqi_data[wqi_col].quantile(0.99)

                # Use the central distribution for visualization
                wqi_plot_data = wqi_data[
                    wqi_data[wqi_col] <= wqi_upper
                ]

                fig = px.histogram(
                    wqi_plot_data,
                    x=wqi_col,
                    color=label_col if label_col else None,
                    nbins=60,
                    title="WQI Score Distribution (99th Percentile View)",
                    color_discrete_map=QUALITY_COLORS
                )

                fig.update_layout(
                    height=450,
                    bargap=0.03,
                    template="plotly_white",
                    xaxis_title="WQI Score",
                    yaxis_title="Number of Records",
                    legend_title_text="Quality Class"
                )

                fig.update_xaxes(showgrid=True)
                fig.update_yaxes(rangemode="tozero", showgrid=True)

                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    f"Showing {len(wqi_plot_data):,} of "
                    f"{len(wqi_data):,} observations. "
                    f"Display limit: {wqi_upper:,.2f}. "
                    "Extreme values remain in the original dataset."
                )
            else:
                st.warning("No valid WQI scores available.")

        st.caption("Quality categories are shown from the uploaded labels where available. If generated from WQI, the app uses illustrative conventional bands; confirm the exact thresholds used in your research notebook.")

    with tab3:
        st.subheader("Basic data structures in action")
        st.markdown("These are implemented in Python and connected to filtered project records.")
        # List
        st.markdown("**1. List / array — selected numeric values**")
        st.code(f"Python list = filtered['{parameter}'].dropna().head(10).tolist()")
        st.write(filtered[parameter].dropna().head(10).tolist())
        # Stack
        st.markdown("**2. Stack (LIFO) — latest viewed record**")
        if "ds_stack" not in st.session_state: st.session_state.ds_stack=[]
        if st.button("Push first filtered record onto stack"):
            if len(filtered): st.session_state.ds_stack.append(filtered.iloc[0].to_dict())
        b1,b2=st.columns(2)
        with b1:
            if st.button("Pop stack (remove latest)"):
                if st.session_state.ds_stack: st.session_state.ds_stack.pop()
        with b2: st.write(f"Stack size: {len(st.session_state.ds_stack)}")
        if st.session_state.ds_stack: st.json({k:str(v) for k,v in st.session_state.ds_stack[-1].items()})
        # Queue
        st.markdown("**3. Queue (FIFO) — records waiting for review**")
        if "ds_queue" not in st.session_state: st.session_state.ds_queue=deque()
        q1,q2=st.columns(2)
        with q1:
            if st.button("Enqueue first 5 records"):
                for idx,row in filtered.head(5).iterrows(): st.session_state.ds_queue.append({"row_index":str(idx),parameter:row[parameter]})
        with q2:
            if st.button("Dequeue next record"):
                if st.session_state.ds_queue: st.session_state.ds_queue.popleft()
        st.write(f"Queue length: {len(st.session_state.ds_queue)}")
        st.dataframe(pd.DataFrame(list(st.session_state.ds_queue)),use_container_width=True,hide_index=True)
        # Hash map
        st.markdown("**4. Hash map / dictionary — quality class counts in average O(1) lookup**")
        class_map=filtered[label_col].astype(str).value_counts().to_dict() if label_col else filtered[parameter].describe().to_dict()
        lookup_key=st.selectbox("Look up a key",list(class_map.keys()))
        st.metric(f"Value for {lookup_key}",f"{class_map[lookup_key]:,}" if isinstance(class_map[lookup_key],(int,np.integer)) else str(class_map[lookup_key]))
        # binary search
        st.markdown("**5. Binary search — find a parameter value in sorted unique values**")
        sorted_values=sorted(filtered[parameter].dropna().round(3).unique().tolist())
        target=st.number_input("Target value (rounded to 3 decimals)",value=float(sorted_values[len(sorted_values)//2]) if sorted_values else 0.0)
        if st.button("Binary search target"):
            pos=binary_search(sorted_values,round(target,3))
            st.success(f"Found at sorted index {pos}." if pos>=0 else "Exact rounded value not found in filtered data.")

    with tab4:
        st.subheader("Advanced data structures")
        st.markdown("**1. Binary Search Tree (BST) index** — inserts distinct parameter values and traverses them in sorted order.")
        if st.button("Build BST for selected parameter"):
            tree=BinarySearchTree()
            for idx,val in filtered[parameter].items(): tree.insert(val,{"row_index":str(idx),"value":float(val)})
            ordered=tree.inorder()
            st.session_state["bst_preview"]=[{"row_index":x["row_index"],"value":x["value"]} for x in ordered[:500]]
            st.session_state["bst_nodes"]=tree.count
        if "bst_preview" in st.session_state:
            st.metric("Distinct BST keys",st.session_state.get("bst_nodes",0))
            st.dataframe(pd.DataFrame(st.session_state["bst_preview"]),use_container_width=True,hide_index=True)
        st.divider()
        st.markdown("**2. Priority queue / min-heap** — surface the highest parameter readings for review.")
        if st.button("Build priority queue (top 10 readings)"):
            vals=filtered[parameter].dropna()
            heap=[(-float(v),str(i)) for i,v in vals.items()]
            heapq.heapify(heap)
            top=[]
            for _ in range(min(10,len(heap))):
                neg,idx=heapq.heappop(heap); top.append({"row_index":idx,parameter:-neg})
            st.session_state["heap_top"]=top
        if "heap_top" in st.session_state:
            st.dataframe(pd.DataFrame(st.session_state["heap_top"]),use_container_width=True,hide_index=True)
        st.divider()
        st.markdown("**3. Graph (adjacency list)** — state-to-district links, when both fields exist.")
        if state_col and district_col:
            graph=defaultdict(set)
            for s,d in filtered[[state_col,district_col]].dropna().astype(str).itertuples(index=False,name=None):
                graph[s].add(d)
            graph_df=pd.DataFrame([{"State":s,"District count":len(ds),"Sample districts":", ".join(sorted(ds)[:8])} for s,ds in graph.items()])
            st.metric("State nodes",len(graph))
            st.dataframe(graph_df,use_container_width=True,hide_index=True)
            nodes=[{"id":s,"label":s,"size":10+len(ds)} for s,ds in graph.items()]
            edges=[{"source":s,"target":d} for s,ds in graph.items() for d in list(ds)[:15]]
            if nodes:
                # A readable edge table is more robust than a force-directed graph for large datasets.
                st.caption("Adjacency-list representation: each state node links to its observed district nodes.")
                st.dataframe(pd.DataFrame([{"State":s,"Connected district":d} for s,ds in graph.items() for d in sorted(ds)]),use_container_width=True,hide_index=True)
        else:
            st.info("Graph view requires columns named State and District (or equivalent).")
        st.divider()
        st.markdown("**4. Sorted index / range query**")
        lo,hi=st.columns(2)
        with lo: lower=st.number_input("Minimum value",value=float(filtered[parameter].min()) if filtered[parameter].notna().any() else 0.0)
        with hi: upper=st.number_input("Maximum value",value=float(filtered[parameter].max()) if filtered[parameter].notna().any() else 0.0)
        if st.button("Run range query"):
            result=filtered[filtered[parameter].between(min(lower,upper),max(lower,upper))]
            st.write(f"{len(result):,} records in range.")
            st.dataframe(result.head(500),use_container_width=True,hide_index=True)

    with tab5:
        st.subheader("Filtered data")
        st.write(f"Showing {len(filtered):,} rows and {len(filtered.columns)} columns.")
        st.dataframe(filtered.head(1000),use_container_width=True,hide_index=True)
        csv=filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV",data=csv,file_name="groundwater_filtered.csv",mime="text/csv")
        summary=filtered.describe(include="all").transpose().reset_index(names="Column")
        st.download_button("Download summary statistics",data=summary.to_csv(index=False).encode("utf-8"),
        file_name="groundwater_summary.csv",mime="text/csv")
        st.subheader("Data structure complexity guide")
        st.dataframe(pd.DataFrame([
        ["List / array","Index access","O(1)","Display numeric observations"],
        ["Stack","Push / pop","O(1) amortized","Recently viewed records"],
        ["Queue","Enqueue / dequeue","O(1)","Review workflow"],
        ["Hash map","Lookup","O(1) average","Quality class counts"],
        ["Binary search","Search sorted array","O(log n)","Find an exact parameter value"],
        ["BST","Insert / in-order traversal","O(log n) average; O(n) worst","Sorted parameter index"],
        ["Heap / priority queue","Push / pop max","O(log n)","Prioritize high readings"],
        ["Graph adjacency list","Neighbor traversal","O(V+E) traversal","State-district relationship"],
        ],columns=["Structure","Operation","Typical complexity","Dashboard use"]),use_container_width=True,hide_index=True)

    st.caption("Research note: this dashboard visualizes the dataset and demonstrates data structures; it does not retrain or validate the notebook's ML models. Do not interpret demo data as CGWB observations.")
