// src/components/KeywordInput.js
import React, { useState } from "react";
import '../index.css';
import SubmitButton from "./SubmitButton";
import { FormContainer, InputField, KeywordInputForm, TextElement, DownloadButton } from "./Styles/Container/KeywordInput.style";
import ProgressBarComponent from "./ProgressBarComponent"; // ProgressBarComponent 임포트

const KeywordInput = () => {
  const [keyword, setKeyword] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDownloadable, setIsDownloadable] = useState(false);
  const [videoUrl, setVideoUrl] = useState(null); // 다운로드 링크

  const handleInputChange = (e) => {
    setKeyword(e.target.value);
  };

  const handleSubmit = async (setLoading) => {
    if (keyword.trim() === "") {
      alert("키워드를 입력하세요!");
      return;
    }
    setIsProcessing(true);
    setLoading(true);

    try {
      await new Promise((resolve) => setTimeout(resolve, 3000)); // 백엔드 요청 대기 (가정)
      console.log("백엔드 응답 완료: ", keyword);

      // 백엔드에서 동영상 URL을 받아오는 예시
      const receivedVideoUrl = "https://example.com/path/to/video.mp4"; // 실제 백엔드에서 받은 URL로 대체
      setVideoUrl(receivedVideoUrl);
    } catch (error) {
      console.error("에러 발생: ", error);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (videoUrl) {
      // URL을 사용하여 자동 다운로드
      const link = document.createElement("a");
      link.href = videoUrl;
      link.download = "video.mp4"; // 파일명 지정
      link.click();
    }
  };

  const handleProgressComplete = () => {
    setIsDownloadable(true); // 진행바 완료 후 다운로드 버튼 활성화
    setIsProcessing(false); // 진행바 숨기기
  };

  return (
    <KeywordInputForm>
      <TextElement>최신 뉴스와 관련된 키워드를 입력하고, 요약 영상을 만들어보세요!</TextElement>
      <FormContainer>
        <InputField
          type="text"
          placeholder="키워드를 입력하세요"
          value={keyword}
          onChange={handleInputChange}
        />
        <SubmitButton onSubmit={handleSubmit} />
      </FormContainer>
      {isProcessing && <ProgressBarComponent onComplete={handleProgressComplete} />} {/* ProgressBarComponent로 진행바 관리 */}
      {isDownloadable && (
        <DownloadButton onClick={handleDownload}>다운로드</DownloadButton>
      )}
    </KeywordInputForm>
  );
};

export default KeywordInput;