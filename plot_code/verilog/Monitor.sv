module Monitor (
    input clk,
    input rst_n,
    input [31:0] X_check,
    input [31:0] Y_check,
    output reg [5:0] count_X,  // 0~32 -> 6 bit
    output reg [5:0] count_Y
);
    integer i;
    reg [5:0] temp_count_X;
    reg [5:0] temp_count_Y;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count_X <= 0;
            count_Y <= 0;
        end else begin
            // 统计X_check中1的个数
            temp_count_X = 0;
            temp_count_Y = 0;
            for (i=0; i<32; i=i+1) begin
                temp_count_X = temp_count_X + X_check[i];
                temp_count_Y = temp_count_Y + Y_check[i];
            end
            // 一拍延迟输出
            count_X <= temp_count_X;
            count_Y <= temp_count_Y;
        end
    end
endmodule
