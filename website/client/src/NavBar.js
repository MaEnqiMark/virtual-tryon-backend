
const NavBar = () => {

    return (
        <div style={{
            width: "100%",
            backgroundColor: "darkgreen",
            padding: "10px 20px",
            boxSizing: "border-box"
        }}>
            <ul style={{
                listStyle: "none",
                display: "flex",
                margin: 0,
                padding: 0,
                gap: "20px", // space between items
            }}>
                <li>
                    <a 
                        href="/tryon"
                        style={{
                        color: "white",
                        textDecoration: "none",
                        fontWeight: "bold",
                        padding: "8px 12px",
                        borderRadius: "4px",
                        }}
                        onMouseEnter={(e) => e.target.style.backgroundColor = "green"}
                        onMouseLeave={(e) => e.target.style.backgroundColor = "transparent"}
                    >
                        Try On
                    </a>
                </li>
                <li>
                    <a 
                        href="/suggestion"
                        style={{
                        color: "white",
                        textDecoration: "none",
                        fontWeight: "bold",
                        padding: "8px 12px",
                        borderRadius: "4px",
                        }}
                        onMouseEnter={(e) => e.target.style.backgroundColor = "green"}
                        onMouseLeave={(e) => e.target.style.backgroundColor = "transparent"}
                    >
                        Suggestion
                    </a>
                </li>
                {/* Uncomment for Help link */}
                {/* <li>
                <a style={{ color: "white", textDecoration: "none" }}>Help</a>
                </li> */}
            </ul>
        </div>
    )
}

export default NavBar