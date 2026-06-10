module Scheduler (
    input  clk,
    input  rst_n,
    input  input_valid,
    input  [31:0] X_check,
    input  [31:0] Y_check,
    input  send,
    output reg [47:0] out_pair,  // 每周期输出4个pair，每个pair 12bit -> 4*12=48bit
    output reg out_valid
);
    // ---------- Buffer ----------
    localparam BUFFER_DEPTH = 640;
    reg [11:0] buffer [0:BUFFER_DEPTH-1]; // 每个pair 12bit {x_idx[5:0], y_idx[5:0]}
    reg [9:0] write_ptr, read_ptr;        // 0~639
    reg [9:0] count;                       // buffer中pair数量

    // ---------- 扫描状态 ----------
    reg [5:0] scan_x;
    reg [5:0] scan_y; // 每次扫描Y的4位
    reg scanning;
    integer i;
    reg [3:0] y_mask;
    reg [11:0] temp_pair [0:3];

    // ---------- 主逻辑 ----------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_ptr <= 0; read_ptr <= 0; count <= 0;
            scan_x <= 0; scan_y <= 0; scanning <= 0;
            out_pair <= 0; out_valid <= 0;
        end else begin
            // -------- 扫描输入 --------
            if (input_valid && !scanning) begin
                scanning <= 1;
                scan_x <= 0; scan_y <= 0;
            end

            if (scanning) begin
                if (scan_x < 32) begin
                    if (X_check[scan_x]) begin
                        // 并行扫描Y的4位
                        y_mask = Y_check >> scan_y;
                        for (i=0; i<4; i=i+1) begin
                            if (y_mask[i] && count < BUFFER_DEPTH) begin
                                buffer[write_ptr] <= {scan_x, scan_y+i};
                                write_ptr <= write_ptr + 1;
                                count <= count + 1;
                            end
                        end
                        scan_y <= scan_y + 4;
                        if (scan_y >= 32) begin
                            scan_y <= 0;
                            scan_x <= scan_x + 1;
                        end
                    end else begin
                        scan_x <= scan_x + 1;
                        scan_y <= 0;
                    end
                end else begin
                    scanning <= 0; // 扫描结束
                end
            end

            // -------- 输出阶段 --------
            if (send) begin
                out_valid <= (count >= 4);
                for (i=0; i<4; i=i+1) begin
                    if (count > 0) begin
                        temp_pair[i] <= buffer[read_ptr];
                        read_ptr <= read_ptr + 1;
                        count <= count - 1;
                    end else begin
                        temp_pair[i] <= 12'b0;
                    end
                end
                out_pair <= {temp_pair[3], temp_pair[2], temp_pair[1], temp_pair[0]};
            end else begin
                out_valid <= 0;
            end
        end
    end
endmodule
